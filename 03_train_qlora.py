"""
03_train_qlora.py
=================
Desktop'a kaydedilmiş HuggingFace dataset'ini yükler,
Llama 3.1 70B modelini 4-bit NF4 + LoRA ile eğitir.

Çalıştır:
    source ~/venvs/qlora_lse/bin/activate
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
    python 03_train_qlora.py

Girdi:
    ~/Desktop/qlora_lse_project/lse_hf_dataset/   ← 02_build_dataset.py çıktısı

Çıktı:
    ~/Desktop/qlora_lse_project/llama31_70b_lse_qlora/
        ├── checkpoints/        ← Ara checkpoint'ler
        ├── final_adapter/      ← Eğitilmiş LoRA ağırlıkları
        ├── plots/
        │   ├── training_loss.png
        │   └── loss_heatmap.png
        └── loss_history.json
"""

import os
import sys
import math
import time
import warnings
from pathlib import Path

import torch
import transformers
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# ── Proje modülleri ──────────────────────────────────────────
from config import CFG, DESKTOP
from utils.vram import print_vram, assert_vram_available
from utils.loss_tracker import LossTracker
from utils.plotting import plot_training_losses, plot_loss_heatmap

# ── HuggingFace ───────────────────────────────────────────────
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_warning()

console = Console()

# Çıktı klasörleri
OUT      = Path(CFG["output_dir"])
CKPT_DIR = OUT / "checkpoints"
PLOT_DIR = OUT / "plots"
for d in [OUT, CKPT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  1. SİSTEM KONTROLÜ
# ══════════════════════════════════════════════════════════════

def check_system():
    console.print(Panel.fit(
        "[bold cyan]QLoRA Fine-Tuning — Adım 3/3[/bold cyan]\n"
        "[bold]Meta-Llama-3.1-70B × LSE Financial Dataset[/bold]\n"
        f"[dim]Başlangıç: {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        title="🚀 Başlatılıyor"
    ))

    t = Table(title="Sistem Bilgisi", show_header=True)
    t.add_column("Bileşen",   style="cyan",  width=20)
    t.add_column("Değer",     style="green", width=40)
    t.add_row("Python",       sys.version.split()[0])
    t.add_row("PyTorch",      torch.__version__)
    t.add_row("Transformers", transformers.__version__)
    t.add_row("CUDA",         torch.version.cuda if torch.cuda.is_available() else "YOK")
    t.add_row("BF16 desteği", str(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else "—")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            t.add_row(f"GPU [{i}]", f"{p.name}  ({p.total_memory/1e9:.1f} GB VRAM)")
    console.print(t)

    # VRAM yeterlilik kontrolü (~42 GB gerekli)
    try:
        assert_vram_available(required_gb=40.0)
    except RuntimeError as e:
        console.print(f"[bold red]HATA:[/bold red] {e}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  2. DATASET YÜKLEME
# ══════════════════════════════════════════════════════════════

def load_dataset():
    ds_path = CFG["dataset_path"]
    if not Path(ds_path).exists():
        console.print(
            f"[bold red]HATA:[/bold red] Dataset bulunamadı: {ds_path}\n"
            f"Önce [bold]python 02_build_dataset.py[/bold] çalıştırın."
        )
        sys.exit(1)

    console.print(f"\n[yellow]📂 Dataset yükleniyor → {ds_path}[/yellow]")
    ds = load_from_disk(ds_path)

    t = Table(title="Dataset Özeti")
    t.add_column("Split",          style="cyan")
    t.add_column("Örnek Sayısı",   style="green")
    t.add_column("Ort. Token",     style="yellow")
    for split in ds:
        tok_counts = ds[split]["token_count"]
        avg        = sum(tok_counts) / len(tok_counts) if tok_counts else 0
        t.add_row(split, f"{len(ds[split]):,}", f"{avg:.0f}")
    console.print(t)

    return ds


# ══════════════════════════════════════════════════════════════
#  3. MODEL + TOKENIZER YÜKLEME (4-bit NF4)
# ══════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    console.print("\n[yellow]🤖 Baz model yükleniyor (4-bit NF4)...[/yellow]")

    token = CFG["hf_token"] or os.environ.get("HF_TOKEN", "")
    if not token:
        console.print("[bold red]UYARI: HF_TOKEN ayarlanmamış![/bold red]")

    # ── BitsAndBytes 4-bit yapılandırması ────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",           # Normal Float 4 — optimal kalite
        bnb_4bit_compute_dtype    = torch.bfloat16,  # Ada Lovelace BF16 destekli
        bnb_4bit_use_double_quant = True,            # İç içe nicemleme (ek ~0.4 bit tasarruf)
    )

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["model_id"],
        token             = token,
        trust_remote_code = True,
        padding_side      = "right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    console.print("  [green]✓[/green] Tokenizer yüklendi.")

    # ── Model ────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        quantization_config = bnb_config,
        device_map          = "auto",           # Çoklu GPU desteği
        token               = token,
        trust_remote_code   = True,
        torch_dtype         = torch.bfloat16,
        attn_implementation = "flash_attention_2",  # RTX Ada — Flash Attention 2
    )
    console.print("  [green]✓[/green] Model 4-bit NF4 olarak yüklendi.")
    print_vram("Model yükleme sonrası")

    # ── KBit eğitim hazırlığı ────────────────────────────────
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.config.use_cache = False   # gradient checkpointing ile uyumsuz

    return model, tokenizer


# ══════════════════════════════════════════════════════════════
#  4. LORA ADAPTÖRÜ EKLEME
# ══════════════════════════════════════════════════════════════

def attach_lora(model):
    console.print("\n[yellow]🔗 LoRA adaptörleri ekleniyor...[/yellow]")

    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = CFG["lora_r"],
        lora_alpha     = CFG["lora_alpha"],
        lora_dropout   = CFG["lora_dropout"],
        target_modules = CFG["lora_targets"],
        bias           = "none",
        inference_mode = False,
    )
    model = get_peft_model(model, lora_config)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t = Table(title="LoRA Parametre Özeti")
    t.add_column("Metrik",    style="cyan",        width=30)
    t.add_column("Değer",     style="bold green",  width=20)
    t.add_row("Toplam parametre",       f"{total:,}")
    t.add_row("Eğitilebilir (LoRA)",    f"{trainable:,}")
    t.add_row("Eğitilebilir oran",      f"%{trainable/total*100:.4f}")
    t.add_row("LoRA rank (r)",          str(CFG["lora_r"]))
    t.add_row("LoRA alpha",             str(CFG["lora_alpha"]))
    t.add_row("LoRA dropout",           str(CFG["lora_dropout"]))
    t.add_row("Hedef katmanlar",        ", ".join(CFG["lora_targets"]))
    console.print(t)

    return model


# ══════════════════════════════════════════════════════════════
#  5. EĞİTİM
# ══════════════════════════════════════════════════════════════

def run_training(model, tokenizer, ds):
    console.print("\n[yellow]🏋️  Eğitim başlıyor...[/yellow]")
    print_vram("Eğitim öncesi")

    tracker = LossTracker()

    # ── SFT Yapılandırması ────────────────────────────────────
    training_args = SFTConfig(
        # Klasörler
        output_dir                  = str(CKPT_DIR),

        # Epoch / adım
        num_train_epochs            = CFG["num_epochs"],
        max_steps                   = -1,

        # Batch & gradyan
        per_device_train_batch_size = CFG["batch_size"],
        per_device_eval_batch_size  = CFG["batch_size"],
        gradient_accumulation_steps = CFG["grad_accumulation"],
        gradient_checkpointing      = True,

        # Optimizer
        learning_rate               = CFG["lr"],
        lr_scheduler_type           = CFG["lr_scheduler"],
        warmup_ratio                = CFG["warmup_ratio"],
        weight_decay                = CFG["weight_decay"],
        max_grad_norm               = CFG["max_grad_norm"],
        optim                       = "paged_adamw_8bit",   # VRAM tasarrufu

        # Hassasiyet
        bf16                        = CFG["bf16"],
        fp16                        = CFG["fp16"],

        # Logging / eval / kayıt
        logging_steps               = CFG["logging_steps"],
        evaluation_strategy         = "steps",
        eval_steps                  = CFG["eval_steps"],
        save_strategy               = "steps",
        save_steps                  = CFG["save_steps"],
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,

        # Diğer
        report_to                   = "none",
        seed                        = CFG["seed"],
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,

        # SFT özel
        max_seq_length              = CFG["max_seq_length"],
        dataset_text_field          = "text",
        packing                     = False,
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        args            = training_args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["validation"],
        callbacks       = [
            tracker,
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    # ── Eğitim özeti ─────────────────────────────────────────
    eff_batch = CFG["batch_size"] * CFG["grad_accumulation"]
    console.print(
        f"\n  [dim]Efektif batch  : {eff_batch}  "
        f"({CFG['batch_size']} × {CFG['grad_accumulation']} grad. akümülasyon)\n"
        f"  Optimizer       : paged_adamw_8bit\n"
        f"  LR scheduler    : {CFG['lr_scheduler']}  (warmup %{CFG['warmup_ratio']*100:.0f})\n"
        f"  Precision       : {'BF16' if CFG['bf16'] else 'FP16'}[/dim]\n"
    )

    t0           = time.time()
    train_result = trainer.train()
    elapsed      = time.time() - t0

    # ── Sonuç tablosu ─────────────────────────────────────────
    t = Table(title="Eğitim Sonuçları", show_header=True)
    t.add_column("Metrik",    style="cyan",        width=25)
    t.add_column("Değer",     style="bold green",  width=25)
    t.add_row("Toplam süre",     f"{elapsed/60:.1f} dakika")
    t.add_row("Toplam adım",     str(train_result.global_step))
    t.add_row("Son eğitim kaybı", f"{train_result.training_loss:.4f}")
    t.add_row("Son eğitim PPL",   f"{math.exp(min(train_result.training_loss, 20)):.2f}")
    smry = tracker.summary()
    if smry.get("best_eval_loss"):
        t.add_row("En iyi eval kaybı",  f"{smry['best_eval_loss']:.4f}")
        t.add_row("En iyi eval PPL",    f"{smry['best_eval_ppl']:.2f}")
        t.add_row("En iyi eval adımı",  str(smry["best_eval_step"]))
    console.print(t)
    print_vram("Eğitim sonrası")

    return trainer, tracker, elapsed


# ══════════════════════════════════════════════════════════════
#  6. KAYDET VE GÖRSELLEŞTİR
# ══════════════════════════════════════════════════════════════

def save_and_visualize(trainer, tracker, elapsed, tokenizer):
    # ── LoRA adaptörlerini kaydet ─────────────────────────────
    adapter_path = str(OUT / "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    console.print(f"\n  [green]✓[/green] LoRA adaptörleri → {adapter_path}")

    # ── Kayıp JSON ────────────────────────────────────────────
    tracker.save_history(OUT)

    # ── Grafikler ─────────────────────────────────────────────
    console.print("\n[yellow]📊 Grafikler oluşturuluyor...[/yellow]")
    plot_training_losses(
        tracker.train_losses,
        tracker.eval_losses,
        tracker.lr_history,
        save_dir=PLOT_DIR,
    )
    plot_loss_heatmap(tracker.train_losses, save_dir=PLOT_DIR)

    # ── Son değerlendirme ─────────────────────────────────────
    console.print("\n[yellow]📊 Son doğrulama değerlendirmesi...[/yellow]")
    metrics   = trainer.evaluate()
    final_loss = metrics.get("eval_loss", float("nan"))
    final_ppl  = math.exp(min(final_loss, 20))
    console.print(
        f"  eval_loss = [bold red]{final_loss:.4f}[/bold red]   "
        f"perplexity = [bold yellow]{final_ppl:.2f}[/bold yellow]"
    )

    # ── Final panel ───────────────────────────────────────────
    console.print(Panel.fit(
        f"[bold green]✅ QLoRA Eğitimi Tamamlandı![/bold green]\n\n"
        f"Model      : {CFG['model_id']}\n"
        f"Süre       : {elapsed/60:.1f} dakika\n"
        f"Adaptörler : [cyan]{OUT / 'final_adapter'}[/cyan]\n"
        f"Grafikler  : [cyan]{PLOT_DIR}[/cyan]\n"
        f"Kayıp JSON : [cyan]{OUT / 'loss_history.json'}[/cyan]\n\n"
        f"[bold]Modeli yeniden yüklemek için:[/bold]\n"
        f"  [cyan]from peft import PeftModel\n"
        f"  model = PeftModel.from_pretrained(\n"
        f"      base_model, '{OUT / 'final_adapter'}'\n"
        f"  )[/cyan]",
        title="🎉 Tamamlandı"
    ))


# ══════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════

def main():
    import torch; torch.manual_seed(CFG["seed"])

    check_system()
    ds                    = load_dataset()
    model, tokenizer      = load_model_and_tokenizer()
    model                 = attach_lora(model)
    trainer, tracker, ela = run_training(model, tokenizer, ds)
    save_and_visualize(trainer, tracker, ela, tokenizer)


if __name__ == "__main__":
    main()
