"""
utils/vram.py
=============
GPU VRAM kullanımını izlemek ve raporlamak için yardımcı fonksiyonlar.
"""

import torch
from rich.console import Console
from rich.table import Table

console = Console()


def print_vram(label: str = "") -> None:
    """Tüm GPU'ların mevcut VRAM kullanımını konsola yaz."""
    if not torch.cuda.is_available():
        console.print("[dim]VRAM: CUDA GPU yok.[/dim]")
        return

    t = Table(title=f"VRAM Durumu — {label}" if label else "VRAM Durumu",
              show_header=True, show_edge=False)
    t.add_column("GPU", style="cyan",   width=6)
    t.add_column("Ad",  style="white",  width=28)
    t.add_column("Kullanılan", style="bold green",  width=12)
    t.add_column("Rezerve",    style="yellow",       width=12)
    t.add_column("Toplam",     style="dim",          width=12)
    t.add_column("% Dolu",     style="bold red",     width=10)

    for i in range(torch.cuda.device_count()):
        p       = torch.cuda.get_device_properties(i)
        alloc   = torch.cuda.memory_allocated(i)   / 1e9
        rsvd    = torch.cuda.memory_reserved(i)    / 1e9
        total   = p.total_memory                   / 1e9
        pct     = alloc / total * 100
        t.add_row(
            str(i),
            p.name,
            f"{alloc:.2f} GB",
            f"{rsvd:.2f} GB",
            f"{total:.2f} GB",
            f"%{pct:.1f}",
        )

    console.print(t)


def get_vram_gb(device: int = 0) -> tuple[float, float]:
    """(kullanılan_gb, toplam_gb) döndür."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used  = torch.cuda.memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    return used, total


def assert_vram_available(required_gb: float = 40.0) -> None:
    """Yeterli VRAM yoksa RuntimeError fırlat."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı.")
    _, total = get_vram_gb(0)
    if total < required_gb:
        raise RuntimeError(
            f"Yetersiz VRAM: {total:.1f} GB mevcut, "
            f"en az {required_gb:.0f} GB gerekli."
        )
