"""Keep eval from saturating all CPU cores (laptop-friendly defaults)."""

from __future__ import annotations

import os


def apply_cpu_limits(*, threads: int = 1) -> None:
    """Call once at process start (CLI, tests, benchmark scripts)."""
    n = str(max(1, threads))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", n)
    try:
        import torch

        torch.set_num_threads(int(n))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(int(n))
    except ImportError:
        pass
