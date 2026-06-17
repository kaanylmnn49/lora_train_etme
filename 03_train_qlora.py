import sys
import subprocess
from pathlib import Path
from config import CFG

# Ayarlar
DESKTOP = Path.home() / "Desktop" / "lora_train_etme"
ADAPTER_PATH = DESKTOP / "llama31_8b_lse_mlx" / "final_adapter"

def main():
    # Komutu oluştur
    command = [
        "python", "-m", "mlx_lm.lora",
        "--model", CFG["model_id"],
        "--train",
        "--data", str(DESKTOP), # .jsonl dosyaları burada
        "--iters", str(CFG["iters"]),
        "--batch-size", str(CFG["batch_size"]),
        "--num-layers", "32", # <-- Hata giderildi: num-layers
        "--learning-rate", str(CFG["learning_rate"]),
        "--adapter-path", str(ADAPTER_PATH)
    ]

    print("Eğitim başlatılıyor...")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()