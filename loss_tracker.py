"""
utils/loss_tracker.py
=====================
HuggingFace Trainer için özel callback:
- Her logging adımında train / eval kayıplarını yakalar
- Renkli terminal çıktısı üretir
- Eğitim geçmişini JSON olarak kaydeder
"""

import json
import math
import time
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from rich.console import Console

console = Console()


class LossTracker(TrainerCallback):
    """
    Eğitim ve doğrulama kayıplarını, perplexity'yi ve öğrenme hızını
    adım adım kaydeden Trainer callback'i.
    """

    def __init__(self):
        self.train_losses: list[dict] = []
        self.eval_losses:  list[dict] = []
        self.lr_history:   list[dict] = []
        self._step_start: float | None = None

    # ── Adım başlangıcında zamanlayıcı kur ───────────────────
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._step_start = time.time()

    # ── Log eventi: train ve lr bilgisini kaydet ──────────────
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if not logs:
            return

        step    = state.global_step
        epoch   = state.epoch or 0.0
        elapsed = time.time() - (self._step_start or time.time())

        # ── Eğitim kaybı ──────────────────────────────────────
        if "loss" in logs:
            loss = logs["loss"]
            ppl  = math.exp(min(loss, 20))
            lr   = logs.get("learning_rate", 0.0)

            self.train_losses.append({
                "step"        : step,
                "epoch"       : round(epoch, 4),
                "loss"        : loss,
                "perplexity"  : round(ppl, 4),
                "sec_per_step": round(elapsed, 2),
            })
            self.lr_history.append({"step": step, "lr": lr})

            # İlerleme çubuğu
            max_steps = state.max_steps or 1
            pct       = step / max_steps
            bar_n     = 28
            filled    = int(bar_n * pct)
            bar       = "█" * filled + "░" * (bar_n - filled)

            console.print(
                f"  [dim]Epoch {epoch:5.2f}[/dim] "
                f"[{bar}] "
                f"Adım [cyan]{step:>5}/{max_steps}[/cyan]  "
                f"loss=[bold green]{loss:.4f}[/bold green]  "
                f"ppl=[bold yellow]{ppl:.2f}[/bold yellow]  "
                f"lr=[magenta]{lr:.2e}[/magenta]  "
                f"[dim]{elapsed:.1f}s/adım[/dim]"
            )

        # ── Doğrulama kaybı ───────────────────────────────────
        if "eval_loss" in logs:
            eval_loss = logs["eval_loss"]
            eval_ppl  = math.exp(min(eval_loss, 20))

            self.eval_losses.append({
                "step"      : step,
                "epoch"     : round(epoch, 4),
                "loss"      : eval_loss,
                "perplexity": round(eval_ppl, 4),
            })

            console.print(
                f"\n  ╔══ EVAL ════════════════════════════════╗\n"
                f"  ║  Adım [cyan]{step:>5}[/cyan]   "
                f"eval_loss=[bold red]{eval_loss:.4f}[/bold red]   "
                f"ppl=[bold yellow]{eval_ppl:.2f}[/bold yellow]     ║\n"
                f"  ╚═════════════════════════════════════════╝\n"
            )

    # ── Eğitim bitişinde JSON kaydet ──────────────────────────
    def save_history(self, output_dir: str | Path) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = str(out / "loss_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_losses": self.train_losses,
                    "eval_losses" : self.eval_losses,
                    "lr_history"  : self.lr_history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        console.print(f"  [green]✓[/green] Kayıp geçmişi JSON → {path}")
        return path

    # ── Özet istatistikler ────────────────────────────────────
    def summary(self) -> dict:
        if not self.train_losses:
            return {}
        best_eval = (
            min(self.eval_losses, key=lambda x: x["loss"])
            if self.eval_losses else {}
        )
        return {
            "final_train_loss"  : self.train_losses[-1]["loss"],
            "final_train_ppl"   : self.train_losses[-1]["perplexity"],
            "best_eval_loss"    : best_eval.get("loss"),
            "best_eval_ppl"     : best_eval.get("perplexity"),
            "best_eval_step"    : best_eval.get("step"),
            "total_logged_steps": len(self.train_losses),
        }
