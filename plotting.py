"""
utils/plotting.py
=================
Eğitim sürecine ait tüm görselleştirme fonksiyonları.
03_train_qlora.py tarafından import edilir.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from rich.console import Console

console = Console()


def plot_training_losses(
    train_losses: list[dict],
    eval_losses:  list[dict],
    lr_history:   list[dict],
    save_dir:     str | Path,
) -> str:
    """
    4 panelli eğitim grafiği:
      (0,0) Eğitim kaybı (ham + düzleştirilmiş)
      (0,1) Eğitim vs Doğrulama kaybı
      (1,0) Perplexity (eğitim + doğrulama)
      (1,1) Öğrenme hızı planı
    """
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        "QLoRA Fine-Tuning — Eğitim Metrikleri\n"
        "Meta-Llama-3.1-70B  ×  LSE Financial Dataset  ×  RTX 6000 Ada",
        fontsize=13, fontweight="bold",
    )

    tr  = pd.DataFrame(train_losses) if train_losses else pd.DataFrame()
    val = pd.DataFrame(eval_losses)  if eval_losses  else pd.DataFrame()
    lr  = pd.DataFrame(lr_history)   if lr_history   else pd.DataFrame()

    # ── (0,0) Ham + Düzleştirilmiş Eğitim Kaybı ─────────────
    ax = axes[0, 0]
    if not tr.empty:
        ax.plot(tr["step"], tr["loss"],
                alpha=0.25, color="#5588ff", linewidth=0.7, label="Ham kayıp")
        win = max(5, len(tr) // 25)
        smt = tr["loss"].rolling(win, min_periods=1).mean()
        ax.plot(tr["step"], smt,
                color="#1144cc", linewidth=2.2, label=f"EMA düzleştirilmiş (k={win})")
        ax.set_title("Eğitim Kaybı (Train Loss)", fontweight="bold")
        ax.set_xlabel("Adım"); ax.set_ylabel("Cross-Entropy Loss")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(fontsize=9)
        _add_epoch_lines(ax, tr)

    # ── (0,1) Eğitim & Doğrulama Kaybı ──────────────────────
    ax = axes[0, 1]
    if not tr.empty:
        ax.plot(tr["step"], tr["loss"].rolling(10, min_periods=1).mean(),
                color="#1144cc", linewidth=2.0, label="Eğitim (EMA-10)")
    if not val.empty:
        ax.plot(val["step"], val["loss"],
                "o-", color="#cc2244", linewidth=2.2,
                markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, label="Doğrulama")
    ax.set_title("Eğitim vs Doğrulama Kaybı", fontweight="bold")
    ax.set_xlabel("Adım"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    if not tr.empty:
        _add_epoch_lines(ax, tr)

    # ── (1,0) Perplexity ─────────────────────────────────────
    ax = axes[1, 0]
    if not tr.empty:
        ppl_tr = tr["loss"].clip(upper=15).apply(math.exp)
        ax.plot(tr["step"], ppl_tr,
                color="#22aa55", linewidth=1.5, alpha=0.7, label="Eğitim PPL")
    if not val.empty:
        ppl_val = val["loss"].clip(upper=15).apply(math.exp)
        ax.plot(val["step"], ppl_val,
                "s--", color="#ff7722", linewidth=2.0,
                markersize=7, markeredgecolor="white",
                label="Doğrulama PPL")
    ax.set_title("Perplexity", fontweight="bold")
    ax.set_xlabel("Adım"); ax.set_ylabel("Perplexity  [exp(loss)]")
    ax.legend(fontsize=9)

    # ── (1,1) Öğrenme Hızı ───────────────────────────────────
    ax = axes[1, 1]
    if not lr.empty:
        ax.plot(lr["step"], lr["lr"],
                color="#aa22cc", linewidth=2.0)
        ax.fill_between(lr["step"], lr["lr"], alpha=0.15, color="#aa22cc")
        ax.set_yscale("log")
        ax.set_title("Öğrenme Hızı Planı (Cosine + Warmup)", fontweight="bold")
        ax.set_xlabel("Adım"); ax.set_ylabel("Learning Rate  [log ölçek]")

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / "training_loss.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓[/green] Eğitim grafiği → {save_path}")
    return save_path


def plot_loss_heatmap(
    train_losses: list[dict],
    save_dir:     str | Path,
) -> str:
    """
    Epoch × Adım ısı haritası: hangi adımlarda kayıp yüksekti?
    """
    if not train_losses:
        return ""

    df = pd.DataFrame(train_losses)
    if "epoch" not in df.columns or df.empty:
        return ""

    df["epoch_int"]  = df["epoch"].apply(lambda x: int(x) + 1)
    df["step_in_ep"] = df.groupby("epoch_int").cumcount()

    pivot = df.pivot_table(
        index="epoch_int", columns="step_in_ep",
        values="loss", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd_r",
        cbar_kws={"label": "Loss"},
        linewidths=0, rasterized=True,
    )
    ax.set_title("Kayıp Isı Haritası  (Epoch × Adım)", fontweight="bold")
    ax.set_xlabel("Adım (epoch içi)"); ax.set_ylabel("Epoch")

    plt.tight_layout()
    save_path = str(Path(save_dir) / "loss_heatmap.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓[/green] Isı haritası → {save_path}")
    return save_path


# ── Yardımcı ─────────────────────────────────────────────────

def _add_epoch_lines(ax, tr: pd.DataFrame):
    """Epoch sınırlarını dikey çizgiyle işaretle."""
    if "epoch" not in tr.columns:
        return
    epochs = tr["epoch"].apply(math.floor).unique()
    for ep in epochs[1:]:
        ep_step = tr.loc[tr["epoch"].apply(math.floor) == ep, "step"].min()
        ax.axvline(ep_step, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(ep_step, ax.get_ylim()[1] * 0.97,
                f"Ep{ep}", fontsize=7, color="gray", ha="center")
