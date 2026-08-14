"""Hardware metadata collection.

Records what a benchmark run actually executed on (spec section 15) --
essential for reproducibility, since RTF/latency numbers are meaningless
without knowing the hardware they were measured on. A number is only
comparable to another number measured on the same (or known) hardware.
"""
from __future__ import annotations

import platform
import subprocess

import psutil

from voice_benchmark.core.models import HardwareMetadata


def _get_gpu_info() -> tuple[str | None, float | None]:
    """Best-effort NVIDIA GPU name + VRAM via nvidia-smi.

    Returns (None, None) if there's no GPU or nvidia-smi isn't available --
    never raises, since "no discrete GPU" is a completely normal, common
    case for this project (most STT models here run fine on CPU), not an
    error condition.
    """
    try:
        output = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        if not output:
            return None, None
        first_line = output.splitlines()[0]
        name, mem_mb = (p.strip() for p in first_line.split(","))
        return name, round(float(mem_mb) / 1024, 2)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, OSError):
        return None, None


def collect_hardware_metadata() -> HardwareMetadata:
    """Snapshot of the current machine's OS, Python version, CPU, RAM, and
    (if present) GPU/VRAM. Cheap to call -- no benchmarking, just inspection."""
    gpu_name, vram_gb = _get_gpu_info()
    return HardwareMetadata(
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu=platform.processor() or platform.machine() or None,
        gpu=gpu_name,
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        vram_gb=vram_gb,
    )
