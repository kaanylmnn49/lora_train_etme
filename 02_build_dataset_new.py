import sys
import pandas as pd
from pathlib import Path
from rich.console import Console
from datasets import Dataset

# Masaüstü yolu (ayarları config.py yerine burada sabitliyoruz)
DESKTOP = Path.home() / "Desktop" / "lora_train_etme"
DESKTOP.mkdir(parents=True, exist_ok=True)
console = Console()

def convert_to_hf_dataset(data):
    rows = []
    for _, row in data.iterrows():
        # Modelin öğrenmesini istediğimiz "Uzman Analist" şablonu
        instruction = f"Hisse: {row['Ticker']} için teknik analiz raporu hazırla. RSI: {row['RSI']:.2f}."
        
        # Modelin vermesini istediğimiz kapsamlı cevap (Buna "Ground Truth" denir)
        response = (
            f"### 1. Fiyat ve Trend Değerlendirmesi\n"
            f"{row['Ticker']} hissesi {row['Close']:.2f} seviyesinde seyrediyor. "
            f"Teknik göstergeler trendin yönü hakkında ipuçları sunuyor.\n\n"
            f"### 2. İndikatör Analizi (RSI)\n"
            f"RSI değeri {row['RSI']:.2f} olarak ölçüldü. Bu, hissenin piyasadaki momentumunu "
            f"{'aşırı alım' if row['RSI'] > 70 else 'nötr' if row['RSI'] > 30 else 'aşırı satım'} "
            f"bölgesinde gösteriyor.\n\n"
            f"### 3. Genel Değerlendirme\n"
            f"Veriler ışığında yatırımcıların dikkatli olması önerilir."
        )
        
        # Llama 3.1 Formatına sokuyoruz
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"Sen bir LSE finansal analistisin.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        )
        rows.append({"text": text})
    return Dataset.from_pandas(pd.DataFrame(rows))

def main():
    csv_path = DESKTOP / "lse_raw_data.csv"
    if not csv_path.exists():
        console.print("[bold red]HATA: lse_raw_data.csv bulunamadı![/bold red]")
        sys.exit(1)

    data = pd.read_csv(csv_path)
    ds = convert_to_hf_dataset(data)

    # 1. Veriyi böl
    split_ds = ds.train_test_split(test_size=0.1)

    # 2. .jsonl olarak kaydet (MLX bunu bekler)
    train_path = DESKTOP / "train.jsonl"
    valid_path = DESKTOP / "valid.jsonl"

    split_ds["train"].to_json(train_path, orient="records", lines=True)
    split_ds["test"].to_json(valid_path, orient="records", lines=True)

    console.print(f"[bold green]✓ Başarılı![/bold green] Dosyalar oluşturuldu:")
    console.print(f"  {train_path}")
    console.print(f"  {valid_path}")

if __name__ == "__main__":
    main()