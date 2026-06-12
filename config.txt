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
    "dataset_path"     : str(DESKTOP), # MLX, train.jsonl dosyasını burada arar
    "output_dir"       : str(DESKTOP / "llama31_8b_lse_mlx"),

    "model_id"         : "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",

    "max_samples"      : 4000,
    "val_split"        : 0.10,
    "max_seq_length"   : 1024,
    "seed"             : 42,

    # M5 Max için Optimize Eğitim Parametreleri
    "lora_layers"      : 32,
    "iters"            : 1000,
    "batch_size"       : 4,
    "learning_rate"    : 2e-5,
    "steps_per_eval"   : 50,
    "val_batches"      : 10,
}