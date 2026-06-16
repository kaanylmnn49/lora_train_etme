"""
02_build_dataset.py
===================
lse_raw_data.csv'yi okur, her satırı detaylı 7 bölümlü teknik analiz
raporuna (instruction + response) dönüştürür, Llama 3.1 chat template'ine
sokar ve train.jsonl / valid.jsonl olarak kaydeder (MLX bunları bekler).

Çalıştır:
    python 02_build_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from datasets import Dataset

DESKTOP = Path.home() / "Desktop" / "lora_train_etme"
DESKTOP.mkdir(parents=True, exist_ok=True)
console = Console()

CFG = {
    "raw_csv_path" : str(DESKTOP / "lse_raw_data.csv"),
    "max_samples"  : 4000,
    "val_split"    : 0.10,
    "seed"         : 42,
}


# ══════════════════════════════════════════════════════════════
#  PROMPT ÜRETİCİ — Her satırdan instruction/response çifti
# ══════════════════════════════════════════════════════════════

def _rsi_comment(rsi: float) -> str:
    if rsi >= 75:   return "güçlü aşırı alım bölgesinde (≥75)"
    if rsi >= 70:   return "aşırı alım bölgesinde (70–75)"
    if rsi <= 25:   return "güçlü aşırı satım bölgesinde (≤25)"
    if rsi <= 30:   return "aşırı satım bölgesinde (25–30)"
    return "nötr bölgede (30–70)"

def _trend_label(row: pd.Series) -> str:
    if pd.isna(row.get("MA50")):
        return "belirsiz"
    return "yükseliş" if float(row["Close"]) > float(row["MA50"]) else "düşüş"

def _macd_label(row: pd.Series) -> str:
    m = float(row["MACD"])
    s = float(row["Signal"])
    if m > s and m > 0:   return "güçlü boğa sinyali (MACD pozitif ve sinyal üzerinde)"
    if m > s and m <= 0:  return "zayıf boğa sinyali (MACD negatif ama sinyal üzerinde)"
    if m < s and m < 0:   return "güçlü ayı sinyali (MACD negatif ve sinyal altında)"
    return "zayıf ayı sinyali (MACD pozitif ama sinyal altında)"

def _vol_comment(vol_ratio: float) -> str:
    if vol_ratio >= 2.0:  return f"ortalamanın {vol_ratio:.1f} katı — anormal yüksek hacim"
    if vol_ratio >= 1.3:  return f"ortalamanın {vol_ratio:.1f} katı — yüksek hacim"
    if vol_ratio <= 0.5:  return f"ortalamanın {vol_ratio:.1f} katı — düşük hacim"
    return f"ortalamanın {vol_ratio:.1f} katı — normal hacim"

def _overall_signal(row: pd.Series) -> str:
    trend  = _trend_label(row)
    rsi    = float(row["RSI"])
    macd_h = float(row["MACD_Hist"]) if "MACD_Hist" in row.index else 0.0
    bulls  = int(trend == "yükseliş") + int(macd_h > 0) + int(30 < rsi < 60)
    if bulls >= 3: return "olumlu"
    if bulls <= 1: return "olumsuz"
    return "karışık"


def row_to_sample(row: pd.Series):
    """Tek bir CSV satırından instruction + response üret."""
    try:
        ticker  = str(row["Ticker"])
        sector  = str(row.get("Sector", "Diğer"))
        date    = pd.to_datetime(row["Date"]).strftime("%d %B %Y")
        price   = float(row["Close"])
        ret1    = float(row["Ret1"]) * 100
        ret5    = float(row["Ret5"]) * 100
        ret20   = float(row["Ret20"]) * 100
        rsi     = float(row["RSI"])
        macd    = float(row["MACD"])
        signal  = float(row["Signal"])
        bb_pos  = float(row["BB_POS"]) * 100
        vol_rat = float(row["VolRatio"])
        atr     = float(row["ATR"]) if "ATR" in row.index else 0.0
        trend   = _trend_label(row)
    except Exception:
        return None

    instruction = (
        f"Aşağıdaki London Stock Exchange (LSE) piyasa verilerini kullanarak "
        f"{ticker} hissesi ({sector} sektörü) için kapsamlı bir teknik analiz raporu hazırla.\n\n"
        f"=== Piyasa Verileri ({date}) ===\n"
        f"Hisse          : {ticker}\n"
        f"Sektör         : {sector}\n"
        f"Kapanış Fiyatı : {price:.2f} GBX\n"
        f"Günlük Getiri  : {ret1:+.2f}%\n"
        f"5 Günlük Getiri: {ret5:+.2f}%\n"
        f"20 Günlük Getiri: {ret20:+.2f}%\n"
        f"Hacim Durumu   : {_vol_comment(vol_rat)}\n"
        f"RSI(14)        : {rsi:.1f}\n"
        f"MACD           : {macd:.5f}  |  Sinyal: {signal:.5f}\n"
        f"Bollinger Pozisyon: %{bb_pos:.0f}  (0=alt bant, 100=üst bant)\n"
        f"ATR(14)        : {atr:.2f} GBX\n"
        f"50G MA Trend   : {trend}\n"
    )

    response = (
        f"## {ticker} Teknik Analiz Raporu — {date}\n\n"
        f"### 1. Fiyat ve Trend Değerlendirmesi\n"
        f"{ticker} hissesi {date} itibarıyla {price:.2f} GBX kapanış yapmıştır. "
        f"Hisse, 50 günlük hareketli ortalamaya göre **{trend} trendi** içindedir. "
        f"Günlük bazda {ret1:+.2f}%, 5 günlük bazda {ret5:+.2f}% ve son 20 günde "
        f"{ret20:+.2f}% getiri sağlamıştır.\n\n"
        f"### 2. Momentum Analizi (RSI)\n"
        f"RSI(14) değeri **{rsi:.1f}** olup {_rsi_comment(rsi)}. "
        f"{'Kısa vadeli geri çekilme riski göz ardı edilmemelidir.' if rsi > 65 else 'Momentum açısından belirgin bir aşırılık bulunmamaktadır.' if rsi >= 35 else 'Teknik olarak toparlanma potansiyeli mevcuttur.'}\n\n"
        f"### 3. MACD Sinyali\n"
        f"Mevcut durum **{_macd_label(row)}** olarak değerlendirilebilir. "
        f"MACD({macd:.5f}) ve sinyal hattı({signal:.5f}) karşılaştırması "
        f"{'alım yönünde baskı olduğuna işaret etmektedir.' if macd > signal else 'satış yönünde baskı olduğuna işaret etmektedir.'}\n\n"
        f"### 4. Bollinger Bantları\n"
        f"Fiyat, bantların **%{bb_pos:.0f}** konumunda işlem görmektedir. "
        f"{'Üst banda yakınlık aşırı alım sinyali olarak yorumlanabilir.' if bb_pos > 85 else 'Alt banda yakınlık olası bir destek bölgesine işaret etmektedir.' if bb_pos < 15 else 'Bantların orta kesiminde bulunması nötr bir pozisyona işaret etmektedir.'}\n\n"
        f"### 5. Hacim Analizi\n"
        f"{_vol_comment(vol_rat).capitalize()}. "
        f"{'Yüksek hacim, fiyat hareketinin güçlü bir teyidini sağlar.' if vol_rat > 1.3 else 'Düşük hacim, mevcut trendin güç kaybettiğine işaret edebilir.' if vol_rat < 0.7 else 'Normal hacim seviyeleri fiyat hareketini orta düzeyde desteklemektedir.'}\n\n"
        f"### 6. Volatilite (ATR)\n"
        f"14 günlük ATR değeri {atr:.2f} GBX olup bu durum "
        f"{'yüksek volatiliteye' if atr > price * 0.03 else 'düşük volatiliteye' if atr < price * 0.01 else 'orta düzey volatiliteye'} işaret etmektedir. "
        f"Stop-loss seviyeleri belirlenirken bu değer dikkate alınmalıdır.\n\n"
        f"### 7. Genel Değerlendirme\n"
        f"Teknik göstergeler bütünsel olarak **{_overall_signal(row)}** bir görünüm "
        f"sergilemektedir. {sector} sektöründeki makroekonomik gelişmeler ve şirket "
        f"haberleri takip edilmelidir. Bu analiz yatırım tavsiyesi niteliği taşımamaktadır."
    )

    return {"instruction": instruction, "response": response}


# ══════════════════════════════════════════════════════════════
#  LLAMA 3.1 CHAT TEMPLATE
# ══════════════════════════════════════════════════════════════

def build_chat_text(instruction: str, response: str) -> str:
    system_prompt = (
        "Sen deneyimli bir London Stock Exchange (LSE) finansal analistisin. "
        "Teknik analiz, temel analiz ve piyasa psikolojisi konularında uzmansın. "
        "Veriye dayalı, nesnel ve kapsamlı değerlendirmeler yaparsın."
    )
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )


# ══════════════════════════════════════════════════════════════
#  DATASET DÖNÜŞÜMÜ
# ══════════════════════════════════════════════════════════════

def convert_to_hf_dataset(data: pd.DataFrame) -> Dataset:
    console.print("\n[yellow]🔧 Satırlar prompt formatına dönüştürülüyor...[/yellow]")

    n_sample = min(CFG["max_samples"], len(data))
    sample_df = data.sample(n=n_sample, random_state=CFG["seed"]).reset_index(drop=True)

    records = []
    skipped = 0

    for _, row in sample_df.iterrows():
        sample = row_to_sample(row)
        if sample is None:
            skipped += 1
            continue

        text = build_chat_text(sample["instruction"], sample["response"])
        records.append({
            "text"  : text,
            "ticker": str(row["Ticker"]),
            "date"  : str(row["Date"])[:10],
        })

    console.print(
        f"  [green]✓[/green] {len(records):,} örnek oluşturuldu  "
        f"[dim](atlandı: {skipped})[/dim]"
    )

    return Dataset.from_pandas(pd.DataFrame(records))


# ══════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════

def main():
    csv_path = Path(CFG["raw_csv_path"])
    if not csv_path.exists():
        console.print("[bold red]HATA: lse_raw_data.csv bulunamadı![/bold red]")
        sys.exit(1)

    data = pd.read_csv(csv_path, parse_dates=["Date"])
    console.print(f"  CSV yüklendi: {len(data):,} satır, {data['Ticker'].nunique()} hisse")

    ds = convert_to_hf_dataset(data)

    # Train / Validation ayırma
    split_ds = ds.train_test_split(test_size=CFG["val_split"], seed=CFG["seed"])

    train_path = DESKTOP / "train.jsonl"
    valid_path = DESKTOP / "valid.jsonl"

    split_ds["train"].to_json(train_path, orient="records", lines=True, force_ascii=False)
    split_ds["test"].to_json(valid_path, orient="records", lines=True, force_ascii=False)

    t = Table(title="Dataset Bölünmesi")
    t.add_column("Split", style="cyan")
    t.add_column("Örnek", style="green")
    t.add_row("Eğitim", f"{len(split_ds['train']):,}")
    t.add_row("Doğrulama", f"{len(split_ds['test']):,}")
    console.print(t)

    console.print(f"\n[bold green]✓ Başarılı![/bold green] Dosyalar oluşturuldu:")
    console.print(f"  {train_path}")
    console.print(f"  {valid_path}")
    console.print(f"\n[bold]Sonraki adım:[/bold] python 03_train_qlora.py")


if __name__ == "__main__":
    main()
