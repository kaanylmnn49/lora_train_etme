"""
02_build_dataset.py
===================
01_download_lse.py'nin Masaüstüne kaydettiği CSV'yi okur,
instruction-tuning formatına dönüştürür ve yine Masaüstüne
HuggingFace Dataset olarak kaydeder.

Çalıştır:
    python 02_build_dataset.py

Girdi:
    ~/Desktop/qlora_lse_project/lse_raw_data.csv

Çıktı:
    ~/Desktop/qlora_lse_project/lse_hf_dataset/   (train + validation split)
    ~/Desktop/qlora_lse_project/plots/token_dist.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from config import CFG, SECTOR_MAP, DESKTOP

warnings.filterwarnings("ignore")
console = Console()

PLOTS_DIR = DESKTOP / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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
    """MA50'ye göre ana trend."""
    if pd.isna(row["MA50"]):
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

def row_to_sample(row: pd.Series) -> dict | None:
    """Tek bir CSV satırından instruction + response üret."""
    try:
        ticker   = str(row["Ticker"])
        sector   = str(row["Sector"])
        date     = pd.to_datetime(row["Date"]).strftime("%d %B %Y")
        price    = float(row["Close"])
        volume   = float(row["Volume"])
        ret1     = float(row["Ret1"]) * 100
        ret5     = float(row["Ret5"]) * 100
        ret20    = float(row["Ret20"]) * 100
        rsi      = float(row["RSI"])
        macd     = float(row["MACD"])
        signal   = float(row["Signal"])
        bb_pos   = float(row["BB_POS"]) * 100
        vol_rat  = float(row["VolRatio"])
        atr      = float(row["ATR"]) if "ATR" in row.index else 0.0
        trend    = _trend_label(row)
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
#  DATASET DÖNÜŞÜMÜ
# ══════════════════════════════════════════════════════════════

def build_chat_text(tokenizer, instruction: str, response: str) -> str | None:
    """Llama-3 chat şablonuna dönüştür, token limitini kontrol et."""
    messages = [
        {
            "role": "system",
            "content": (
                "Sen deneyimli bir London Stock Exchange (LSE) finansal analistisin. "
                "Teknik analiz, temel analiz ve piyasa psikolojisi konularında uzmansın. "
                "Veriye dayalı, nesnel ve kapsamlı değerlendirmeler yaparsın."
            )
        },
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if token_count > CFG["max_seq_length"]:
        return None
    return text


def convert_to_hf_dataset(data: pd.DataFrame, tokenizer) -> DatasetDict:
    console.print("\n[yellow]🔧 Satırlar prompt formatına dönüştürülüyor...[/yellow]")

    rng      = np.random.RandomState(CFG["seed"])
    n_sample = min(CFG["max_samples"], len(data))
    indices  = rng.choice(len(data), n_sample, replace=False)

    records     = []
    skipped_gen = 0
    skipped_tok = 0

    for idx in tqdm(indices, desc="Prompt üretiliyor", unit="satır"):
        row    = data.iloc[idx]
        sample = row_to_sample(row)
        if sample is None:
            skipped_gen += 1
            continue

        text = build_chat_text(tokenizer, sample["instruction"], sample["response"])
        if text is None:
            skipped_tok += 1
            continue

        records.append({
            "text"       : text,
            "ticker"     : str(row["Ticker"]),
            "sector"     : str(row["Sector"]),
            "date"       : str(row["Date"])[:10],
            "token_count": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
        })

    console.print(
        f"  [green]✓[/green] {len(records):,} örnek oluşturuldu  "
        f"[dim](atlandı: üretim={skipped_gen}, token limiti={skipped_tok})[/dim]"
    )

    # Train / Validation ayırma
    df = pd.DataFrame(records).sample(frac=1, random_state=CFG["seed"]).reset_index(drop=True)
    n_val = int(len(df) * CFG["val_split"])
    df_train = df.iloc[n_val:].reset_index(drop=True)
    df_val   = df.iloc[:n_val].reset_index(drop=True)

    t = Table(title="Dataset Bölünmesi")
    t.add_column("Split",   style="cyan")
    t.add_column("Örnek",   style="green")
    t.add_column("Ort. Token", style="yellow")
    t.add_row("Eğitim",      f"{len(df_train):,}", f"{df_train['token_count'].mean():.0f}")
    t.add_row("Doğrulama",   f"{len(df_val):,}",   f"{df_val['token_count'].mean():.0f}")
    t.add_row("TOPLAM",      f"{len(df):,}",        f"{df['token_count'].mean():.0f}")
    console.print(t)

    return DatasetDict({
        "train"     : Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_val),
    })


# ══════════════════════════════════════════════════════════════
#  TOKEN DAĞILIMI GRAFİĞİ
# ══════════════════════════════════════════════════════════════

def plot_token_distribution(ds: DatasetDict):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Token Dağılımı", fontsize=13, fontweight="bold")

    for ax, split in zip(axes, ["train", "validation"]):
        tokens = ds[split]["token_count"]
        ax.hist(tokens, bins=40, color="#5588ff" if split == "train" else "#ff7744",
                edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(tokens), color="red", linestyle="--",
                   label=f"Ortalama: {np.mean(tokens):.0f}")
        ax.axvline(CFG["max_seq_length"], color="black", linestyle=":",
                   label=f"Max limit: {CFG['max_seq_length']}")
        ax.set_title(f"{split.capitalize()} Token Dağılımı")
        ax.set_xlabel("Token Sayısı"); ax.set_ylabel("Frekans")
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = str(PLOTS_DIR / "token_distribution.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓[/green] Token dağılımı grafiği → {save_path}")


# ══════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════

def main():
    # 1. CSV yükle
    csv_path = CFG["raw_csv_path"]
    if not Path(csv_path).exists():
        console.print(
            f"[bold red]HATA:[/bold red] CSV bulunamadı: {csv_path}\n"
            f"Önce [bold]python 01_download_lse.py[/bold] çalıştırın."
        )
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Dataset Oluşturucu[/bold cyan]\n"
        f"Kaynak : [dim]{csv_path}[/dim]",
        title="🔧 Başlıyor"
    ))

    data = pd.read_csv(csv_path, parse_dates=["Date"])
    console.print(f"  CSV yüklendi: {len(data):,} satır, {data['Ticker'].nunique()} hisse")

    # 2. Tokenizer yükle (şablon için)
    console.print("\n[yellow]🔤 Tokenizer yükleniyor...[/yellow]")
    import os
    token = CFG["hf_token"] or os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["model_id"],
        token=token,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    console.print("  Tokenizer hazır.")

    # 3. Dataset oluştur
    ds = convert_to_hf_dataset(data, tokenizer)

    # 4. Masaüstüne kaydet
    ds_path = CFG["dataset_path"]
    ds.save_to_disk(ds_path)
    console.print(f"\n  [bold green]✓ Dataset kaydedildi[/bold green] → {ds_path}")

    # 5. Token dağılımı grafiği
    console.print("\n[yellow]📊 Grafikler oluşturuluyor...[/yellow]")
    plot_token_distribution(ds)

    # 6. Özet
    console.print(Panel.fit(
        f"[bold green]Dataset hazırlama tamamlandı![/bold green]\n\n"
        f"Dataset → [cyan]{ds_path}[/cyan]\n"
        f"Grafik  → [cyan]{PLOTS_DIR / 'token_distribution.png'}[/cyan]\n\n"
        f"Sonraki adım:\n"
        f"  [bold]python 03_train_qlora.py[/bold]",
        title="✅ Bitti"
    ))


if __name__ == "__main__":
    main()
