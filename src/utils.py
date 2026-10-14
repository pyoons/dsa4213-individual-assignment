"""
Shared helpers: seeding, config loading, JSON logs, timing, hardware info.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed every random number generator used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # data prep and the n-gram model do not need torch
        pass


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str | Path) -> Path:
    """Interpret relative paths from the project root, not the shell cwd."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_json(obj, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str | Path):
    with open(resolve(path), encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)}")


class Timer:
    """Context manager: `with Timer() as t: ...; t.seconds`.

    Calls torch.cuda.synchronize() on exit when a GPU is in use so that
    asynchronous kernels are included in the measured time.
    """

    def __init__(self, cuda_sync: bool = False):
        self.cuda_sync = cuda_sync
        self.seconds = None

    def __enter__(self):
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self._start
        return False

    def _sync(self):
        if self.cuda_sync:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()


def hardware_info() -> dict:
    """Describe the machine so timings in the report are interpretable."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["device"] = "Apple MPS"
        else:
            info["device"] = "CPU"
    except ImportError:
        info["device"] = "CPU (torch not installed)"
    return info
