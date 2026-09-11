#!/usr/bin/env python3
"""Colab / GPU env verify (manifest-driven)."""

from __future__ import annotations

import sys
from pathlib import Path

from compat_audit import audit, main as audit_main

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(audit(ROOT / ".python-env-compat.toml", run_pip=True))
    if "--manifest" not in sys.argv:
        sys.argv[1:1] = ["--manifest", str(ROOT / ".python-env-compat.toml")]
    raise SystemExit(audit_main())
