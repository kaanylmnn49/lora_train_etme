"""
config.py
=========
Tüm ayarların merkezi tek yeri.
Diğer tüm dosyalar buradan CFG'yi import eder.
Değiştirmek istediğin tek yer burasıdır.
"""

from pathlib import Path

# ── Masaüstü yolu (Linux / macOS / Windows hepsi çalışır) ─────
DESKTOP = Path.home() / "Desktop" / "qlora_lse_project"
DESKTOP.mkdir(parents=True, exist_ok=True)

CFG = {
    # ── Masaüstü kayıt yolları ────────────────────────────────
    "raw_csv_path"     : str(DESKTOP / "lse_raw_data.csv"),
    "dataset_path"     : str(DESKTOP / "lse_hf_dataset"),    # HuggingFace dataset klasörü
    "output_dir"       : str(DESKTOP / "llama31_70b_lse_qlora"),

    # ── Model ─────────────────────────────────────────────────
    "model_id"         : "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "hf_token"         : "",          # export HF_TOKEN=... veya buraya yaz

    # ── LSE / İndirme ─────────────────────────────────────────
    "tickers"          : [
        "HSBA.L","BP.L","SHEL.L","AZN.L","ULVR.L",
        "GSK.L","LGEN.L","BT-A.L","VOD.L","BATS.L",
        "LLOY.L","BARC.L","NWG.L","STAN.L","ABF.L",
        "REL.L","NG.L","SSE.L","CNA.L","MKS.L",
    ],
    "period_days"      : 365 * 3,     # 3 yıllık geçmiş veri

    # ── Dataset hazırlama ─────────────────────────────────────
    "max_samples"      : 4000,        # eğitime alınacak max örnek
    "val_split"        : 0.10,        # %10 doğrulama seti
    "max_seq_length"   : 1024,        # maksimum token uzunluğu
    "seed"             : 42,

    # ── QLoRA / LoRA ──────────────────────────────────────────
    "lora_r"           : 32,
    "lora_alpha"       : 64,          # genelde 2 × r
    "lora_dropout"     : 0.05,
    "lora_targets"     : [
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],

    # ── Eğitim ────────────────────────────────────────────────
    "num_epochs"       : 3,
    "batch_size"       : 2,           # RTX 6000 Ada → 48 GB
    "grad_accumulation": 8,           # efektif batch = 2 × 8 = 16
    "lr"               : 2e-4,
    "lr_scheduler"     : "cosine",
    "warmup_ratio"     : 0.05,
    "weight_decay"     : 0.001,
    "max_grad_norm"    : 0.3,
    "bf16"             : True,        # RTX Ada bfloat16 destekli
    "fp16"             : False,
    "logging_steps"    : 5,
    "eval_steps"       : 50,
    "save_steps"       : 100,
}

# Sektör haritası — veri indirme ve dataset aşamalarında kullanılır
SECTOR_MAP = {
    "HSBA.L" : "Bankacılık",
    "BP.L"   : "Enerji",
    "SHEL.L" : "Enerji",
    "AZN.L"  : "İlaç",
    "ULVR.L" : "Tüketim",
    "GSK.L"  : "İlaç",
    "LGEN.L" : "Sigorta",
    "BT-A.L" : "Telekomünikasyon",
    "VOD.L"  : "Telekomünikasyon",
    "BATS.L" : "Tüketim",
    "LLOY.L" : "Bankacılık",
    "BARC.L" : "Bankacılık",
    "NWG.L"  : "Bankacılık",
    "STAN.L" : "Bankacılık",
    "ABF.L"  : "Gıda",
    "REL.L"  : "Medya",
    "NG.L"   : "Kamu Hizmetleri",
    "SSE.L"  : "Kamu Hizmetleri",
    "CNA.L"  : "Sigorta",
    "MKS.L"  : "Perakende",
}
