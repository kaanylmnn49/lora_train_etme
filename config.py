"""
config.py
=========
Apple M5 Max (MLX) mimarisine göre optimize edilmiştir.
"""
from pathlib import Path

# Masaüstü yolu
DESKTOP = Path.home() / "Desktop" / "lora_train_etme"
DESKTOP.mkdir(parents=True, exist_ok=True)

CFG = {
    "raw_csv_path"     : str(DESKTOP / "lse_raw_data.csv"),
    "dataset_path"     : str(DESKTOP),  # MLX, train.jsonl dosyasını burada arar
    "output_dir"       : str(DESKTOP / "llama31_8b_lse_mlx"),

    "model_id"         : "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",

    "max_samples"      : 4000,
    "val_split"        : 0.10,
    "max_seq_length"   : 1024,
    "seed"             : 42,

    # ── LoRA Ayarları ────────────────────────────────────────
    "lora_layers"      : 16,
    "lora_rank"        : 16,
    "lora_alpha"       : 32,
    "lora_dropout"     : 0.05,
    "lora_target"      : ["self_attn.q_proj", "self_attn.k_proj"],

    # ── Eğitim Ayarları ──────────────────────────────────────
    "num_epochs"       : 3,
    "batch_size"       : 4,
    "learning_rate"    : 1e-4,
    "steps_per_eval"   : 50,
    "val_batches"      : 10,
}

# ── Epoch -> Iters Hesabı ───────────────────────────────────
train_samples = int(CFG["max_samples"] * (1 - CFG["val_split"]))
steps_per_epoch = max(1, train_samples // CFG["batch_size"])
CFG["iters"] = steps_per_epoch * CFG["num_epochs"]
