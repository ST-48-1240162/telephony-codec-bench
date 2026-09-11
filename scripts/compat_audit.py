#!/usr/bin/env python3
"""Runtime compatibility audit driven by .python-env-compat.toml."""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _parse_mm(version: str) -> tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", version)]
    if len(nums) < 2:
        return (0, 0)
    return (nums[0], nums[1])


def _import_name(mod_name: str) -> str:
    return mod_name.replace("-", "_")


def _ver(mod_name: str) -> str:
    mod = importlib.import_module(_import_name(mod_name))
    return getattr(mod, "__version__", "unknown")


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def _expected_torchvision_mm(torch_mm: tuple[int, int]) -> tuple[int, int]:
    major, minor = torch_mm
    if major == 2:
        return (0, minor + 15)
    return torch_mm


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _major_ok(version: str, max_major: int) -> bool:
    major = int(version.split(".")[0])
    return major <= max_major


def _min_torch_ok(version: str, minimum: str) -> bool:
    return _parse_mm(version.split("+")[0]) >= _parse_mm(minimum)


def run_pip_check() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        print("  [OK] pip check")
        return True
    lines = [ln for ln in out.splitlines() if "noisekit" not in ln.lower()]
    if not lines:
        print("  [OK] pip check (ignored noisekit vs torch metadata on Colab)")
        return True
    print("  [FAIL] pip check")
    for line in lines:
        print(f"         {line}")
    return False


def audit(manifest_path: Path, *, run_pip: bool) -> int:
    if not manifest_path.is_file():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 2

    cfg = _load_manifest(manifest_path)
    meta = cfg.get("meta", {})
    runtime = cfg.get("runtime", {})
    packages = cfg.get("packages", {})
    torch_cfg = cfg.get("torch", {})
    imports = cfg.get("imports", {})

    ok = True
    project = meta.get("project", manifest_path.parent.name)
    print(f"Compat audit: {project}")
    print("Python", sys.version.split()[0])

    if run_pip:
        ok &= run_pip_check()

    import torch

    tv = torch.__version__.split("+")[0]
    torch_mm = _parse_mm(tv)
    min_torch = runtime.get("min_torch")
    if min_torch:
        ok &= _check(f"torch >= {min_torch}", _min_torch_ok(tv, min_torch), tv)

    if runtime.get("requires_cuda"):
        ok &= _check("cuda available", torch.cuda.is_available(), torch.version.cuda or "cpu")

    vision_rule = torch_cfg.get("vision_pairing", "minor_plus_15")
    if vision_rule:
        try:
            import torchvision

            rv = torchvision.__version__.split("+")[0]
            if vision_rule == "minor_plus_15":
                expected = _expected_torchvision_mm(torch_mm)
                ok &= _check(
                    "torch/torchvision pairing",
                    _parse_mm(rv) == expected,
                    f"torch {tv}, torchvision {rv} (expected 0.{expected[1]}.x)",
                )
        except ImportError:
            ok &= _check("torchvision import", False, "missing")

    audio_rule = torch_cfg.get("audio_pairing", "match_torch")
    if audio_rule:
        try:
            import torchaudio

            av = torchaudio.__version__.split("+")[0]
            if audio_rule == "match_torch":
                ok &= _check(
                    "torch/torchaudio major.minor",
                    _parse_mm(av) == torch_mm,
                    f"torch {tv}, torchaudio {av}",
                )
        except ImportError:
            ok &= _check("torchaudio import", False, "missing")

    for name, spec in packages.items():
        if not isinstance(spec, dict):
            continue
        try:
            version = _ver(name)
        except ImportError as exc:
            ok &= _check(f"{name} import", False, str(exc))
            continue

        if "exact" in spec:
            ok &= _check(f"{name} == {spec['exact']}", version == spec["exact"], version)
        if "max_major" in spec:
            ok &= _check(
                f"{name} major <= {spec['max_major']}",
                _major_ok(version, int(spec["max_major"])),
                version,
            )
        if "specifier" in spec:
            ok &= _check(f"{name} ({spec['specifier']})", True, version)

    for pkg in imports.get("required", []):
        try:
            ok &= _check(f"{pkg} import", True, _ver(pkg))
        except ImportError as exc:
            ok &= _check(f"{pkg} import", False, str(exc))

    for entry in imports.get("symbols", []):
        if isinstance(entry, dict):
            module = entry.get("module", "")
            names = entry.get("names", [])
        else:
            continue
        try:
            mod = importlib.import_module(module)
            for sym in names:
                ok &= _check(f"{module}.{sym}", hasattr(mod, sym))
        except ImportError as exc:
            ok &= _check(f"{module} import", False, str(exc))

    if ok:
        print(f"\nEnvironment OK for {project}.")
        return 0

    if runtime.get("requires_cuda"):
        import torch as _torch

        if not _torch.cuda.is_available():
            print(
                "\nNo GPU detected. Colab: Runtime → Change runtime type → T4 GPU, "
                "then Runtime → Restart session and rerun install."
            )
    else:
        print("\nEnvironment check failed. Fix pins and rerun compat audit.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".python-env-compat.toml"),
        help="Path to compatibility manifest (default: .python-env-compat.toml)",
    )
    parser.add_argument(
        "--no-pip-check",
        action="store_true",
        help="Skip python -m pip check",
    )
    args = parser.parse_args(argv)
    return audit(args.manifest.resolve(), run_pip=not args.no_pip_check)


if __name__ == "__main__":
    raise SystemExit(main())
