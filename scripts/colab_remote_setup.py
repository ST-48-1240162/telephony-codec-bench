#!/usr/bin/env python3
"""Remote Colab setup + compat audit (run via: colab exec -s NAME -f this_file)."""

from __future__ import annotations

import os
import subprocess
import sys

REPO = "https://github.com/ST-48-1240162/telephony-codec-bench.git"
WORKDIR = "/content/telephony-codec-bench"
# cu130 may ship torch 2.14 without matching torchaudio; cu128 trio is stable on Colab T4.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def run(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def py_eval(py: str, code: str) -> str:
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def install_torch_stack(py: str) -> None:
    run(
        f"{py} -m pip install -q --force-reinstall "
        f"torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url {TORCH_INDEX}"
    )


def main() -> int:
    py = sys.executable

    if os.path.isdir(WORKDIR):
        run(f"rm -rf {WORKDIR}")
    run(f"git clone --depth 1 --branch main {REPO} {WORKDIR}")
    os.chdir(WORKDIR)
    run("apt-get -qq update && apt-get -qq install -y libsndfile1 ffmpeg")

    run(f"{py} -m pip install -q -r docs/colab-requirements.txt")
    install_torch_stack(py)

    info = py_eval(
        py,
        "import torch; "
        "print('cuda:', torch.cuda.is_available()); "
        "print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('torch:', torch.__version__)",
    )
    print(info)

    run(f"{py} -m pip install -q -e . --no-deps")
    run(
        f'{py} -m pip install -q "numpy==2.2.2" "pyarrow>=19.0.0,<22.0.0" '
        f'"numba>=0.61.0" "fsspec==2025.3.0" --force-reinstall'
    )
    run(f"{py} scripts/verify_colab_env.py --no-pip-check")

    print("\nSetup OK. Workdir:", WORKDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
