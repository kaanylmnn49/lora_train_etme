"""
01_download_lse.py
==================
London Stock Exchange (FTSE 100) verilerini yfinance üzerinden indirir,
teknik göstergeleri hesaplar ve Masaüstüne CSV olarak kaydeder.

Çalıştır:
    python 01_download_lse.py

Çıktı:
    ~/Desktop/qlora_lse_project/lse_raw_data.csv
    ~/Desktop/qlora_lse_project/plots/dataset_stats.png
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from config import CFG, SECTOR_MAP, DESKTOP

warnings.filterwarnings("ignore")
console = Console()

PLOTS_DIR = DESKTOP / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  TEKNİK GÖSTERGE HESAPLAMA
# ══════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tek bir hisse için teknik göstergeleri hesapla."""
    df = df.copy()

    # Hareketli ortalamalar
    df["MA5"]   = df["Close"].rolling(5).mean()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # Bollinger Bantları (20 günlük)
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["MA20"] + 2 * df["STD20"]
    df["BB_LW"] = df["MA20"] - 2 * df["STD20"]
    df["BB_POS"] = (df["Close"] - df["BB_LW"]) / (df["BB_UP"] - df["BB_LW"] + 1e-9)

    # Getiriler
    df["Ret1"]  = df["Close"].pct_change(1)
    df["Ret5"]  = df["Close"].pct_change(5)
    df["Ret20"] = df["Close"].pct_change(20)

    # RSI (14 günlük)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD (12/26/9)
    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Hacim göstergesi
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / (df["VolMA20"] + 1e-9)

    # ATR (Average True Range, 14 günlük)
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df


# ══════════════════════════════════════════════════════════════
#  VERİ İNDİRME
# ══════════════════════════════════════════════════════════════

def download_all_tickers() -> pd.DataFrame:
    """Tüm FTSE 100 hisselerini indir, göstergeleri ekle, birleştir."""
    end   = datetime.today()
    start = end - timedelta(days=CFG["period_days"])
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    console.print(Panel.fit(
        f"[bold cyan]LSE Veri İndirme[/bold cyan]\n"
        f"Dönem : {start_str}  →  {end_str}\n"
        f"Hisse : {len(CFG['tickers'])} FTSE 100 hissesi",
        title="📥 Başlıyor"
    ))

    frames = []
    failed = []

    for ticker in track(CFG["tickers"], description="İndiriliyor..."):
        try:
            raw = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty or len(raw) < 60:
                failed.append((ticker, "Yetersiz veri (<60 gün)"))
                continue

            # MultiIndex gelirse düzleştir
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = compute_indicators(raw)
            df.dropna(inplace=True)

            if len(df) < 30:
                failed.append((ticker, "Gösterge hesabı sonrası yetersiz"))
                continue

            df["Ticker"] = ticker
            df["Sector"] = SECTOR_MAP.get(ticker, "Diğer")
            frames.append(df.reset_index())

            console.print(
                f"  [green]✓[/green] {ticker:<10} "
                f"{len(df):>4} gün  "
                f"[dim]({SECTOR_MAP.get(ticker, 'Diğer')})[/dim]"
            )

        except Exception as e:
            failed.append((ticker, str(e)))
            console.print(f"  [red]✗[/red] {ticker:<10} {e}")

    if not frames:
        console.print("[bold red]HATA: Hiç veri indirilemedi.[/bold red]")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)

    # ── Sonuç tablosu ─────────────────────────────────────────
    t = Table(title="İndirme Özeti")
    t.add_column("Metrik", style="cyan")
    t.add_column("Değer",  style="green")
    t.add_row("Başarılı hisse", str(len(frames)))
    t.add_row("Başarısız",      str(len(failed)))
    t.add_row("Toplam satır",   f"{len(data):,}")
    t.add_row("Tarih aralığı",  f"{data['Date'].min().date()} → {data['Date'].max().date()}")
    t.add_row("Kolon sayısı",   str(len(data.columns)))
    console.print(t)

    if failed:
        console.print("[yellow]Başarısız hisseler:[/yellow]")
        for tk, reason in failed:
            console.print(f"  [dim]{tk}: {reason}[/dim]")

    return data


# ══════════════════════════════════════════════════════════════
#  İSTATİSTİK GRAFİKLERİ
# ══════════════════════════════════════════════════════════════

def plot_dataset_stats(data: pd.DataFrame):
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "LSE Raw Dataset — İstatistiksel Özet\n"
        f"({len(data):,} günlük gözlem  |  {data['Ticker'].nunique()} hisse)",
        fontsize=13, fontweight="bold"
    )

    # (0,0) Hisse başına satır sayısı
    cnt = data.groupby("Ticker").size().sort_values(ascending=False)
    axes[0,0].bar(cnt.index, cnt.values, color="#5588ff", edgecolor="white")
    axes[0,0].set_title("Hisse Başına Gün Sayısı")
    axes[0,0].set_xlabel("Ticker"); axes[0,0].set_ylabel("Gün")
    axes[0,0].tick_params(axis="x", rotation=50, labelsize=7)

    # (0,1) Sektör dağılımı
    sec = data.groupby("Sector").size().sort_values()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sec)))
    axes[0,1].barh(sec.index, sec.values, color=colors, edgecolor="white")
    axes[0,1].set_title("Sektör Dağılımı")
    axes[0,1].set_xlabel("Gün Sayısı")

    # (0,2) RSI dağılımı
    axes[0,2].hist(data["RSI"].dropna(), bins=50, color="#ff7744", edgecolor="white", alpha=0.85)
    axes[0,2].axvline(30, color="blue", linestyle="--", alpha=0.8, label="Aşırı satım (30)")
    axes[0,2].axvline(70, color="red",  linestyle="--", alpha=0.8, label="Aşırı alım (70)")
    axes[0,2].set_title("RSI(14) Dağılımı"); axes[0,2].set_xlabel("RSI")
    axes[0,2].legend(fontsize=8)

    # (1,0) 1 günlük getiri dağılımı
    ret = data["Ret1"].dropna() * 100
    axes[1,0].hist(ret.clip(-10, 10), bins=80, color="#22aa88", edgecolor="white", alpha=0.85)
    axes[1,0].axvline(0, color="black", linestyle="-", alpha=0.5)
    axes[1,0].set_title("1G Getiri Dağılımı (%)"); axes[1,0].set_xlabel("Getiri %")

    # (1,1) MACD histogram dağılımı
    axes[1,1].hist(data["MACD_Hist"].dropna().clip(-5, 5), bins=60,
                   color="#aa55cc", edgecolor="white", alpha=0.85)
    axes[1,1].axvline(0, color="black", linestyle="-", alpha=0.5)
    axes[1,1].set_title("MACD Histogram Dağılımı")

    # (1,2) Hacim oranı (gerçek hacim / 20g ort)
    vr = data["VolRatio"].dropna().clip(0, 5)
    axes[1,2].hist(vr, bins=60, color="#ffaa22", edgecolor="white", alpha=0.85)
    axes[1,2].axvline(1, color="red", linestyle="--", alpha=0.7, label="Ortalama (1×)")
    axes[1,2].set_title("Hacim Oranı (VolRatio)"); axes[1,2].set_xlabel("Hacim / 20g Ort.")
    axes[1,2].legend(fontsize=8)

    plt.tight_layout()
    save_path = str(PLOTS_DIR / "dataset_stats.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓[/green] İstatistik grafiği → {save_path}")


# ══════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════

def main():
    # 1. Veriyi indir
    data = download_all_tickers()

    # 2. CSV olarak Masaüstüne kaydet
    out_path = CFG["raw_csv_path"]
    data.to_csv(out_path, index=False, encoding="utf-8-sig")   # Excel'de Türkçe düzgün görünür
    size_mb = Path(out_path).stat().st_size / 1_048_576
    console.print(f"\n  [bold green]✓ CSV kaydedildi[/bold green] → {out_path}  ({size_mb:.2f} MB)")

    # 3. Grafikler
    console.print("\n[yellow]📊 Grafikler oluşturuluyor...[/yellow]")
    plot_dataset_stats(data)

    # 4. Özet bilgi
    console.print(Panel.fit(
        f"[bold green]Veri indirme tamamlandı![/bold green]\n\n"
        f"CSV    → [cyan]{out_path}[/cyan]\n"
        f"Grafik → [cyan]{PLOTS_DIR / 'dataset_stats.png'}[/cyan]\n\n"
        f"Sonraki adım:\n"
        f"  [bold]python 02_build_dataset.py[/bold]",
        title="✅ Bitti"
    ))


if __name__ == "__main__":
    main()
