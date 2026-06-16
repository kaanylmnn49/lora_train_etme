"""
03_train_qlora.py
==================
config.py'deki ayarlara göre bir lora_config.yaml üretir
ve mlx_lm.lora'yı bu config ile çalıştırır.

Çalıştır:
    python 03_train_qlora.py
"""
import subprocess
from pathlib import Path

import yaml

from config import CFG, DESKTOP

ADAPTER_PATH = Path(CFG["output_dir"]) / "final_adapter"
CONFIG_YAML  = DESKTOP / "lora_config.yaml"


def build_yaml():
    """config.py'deki CFG'den mlx_lm.lora'nın istediği YAML'ı üret."""
    yaml_dict = {
        "model"          : CFG["model_id"],
        "train"          : True,
        "fine_tune_type" : "lora",
        "data"           : str(DESKTOP),
        "seed"           : CFG["seed"],
        "iters"          : CFG["iters"],
        "batch_size"     : CFG["batch_size"],
        "learning_rate"  : CFG["learning_rate"],
        "num_layers"     : CFG["lora_layers"],
        "adapter_path"   : str(ADAPTER_PATH),
        "steps_per_eval" : CFG["steps_per_eval"],
        "val_batches"    : CFG["val_batches"],
        "lora_parameters": {
            "rank"   : CFG["lora_rank"],
            "scale"  : float(CFG["lora_alpha"]) / float(CFG["lora_rank"]),
            "dropout": CFG["lora_dropout"],
            "keys"   : CFG["lora_target"],
        },
    }

    with open(CONFIG_YAML, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"YAML config oluşturuldu -> {CONFIG_YAML}")
    print(f"Toplam iterasyon: {CFG['iters']}  ({CFG['num_epochs']} epoch)")


def main():
    build_yaml()

    command = ["python", "-m", "mlx_lm.lora", "--config", str(CONFIG_YAML)]

    print("\nEğitim başlatılıyor...")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
