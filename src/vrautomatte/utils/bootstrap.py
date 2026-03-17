"""Ensure the correct PyTorch variant is installed for the current hardware.

Runs before any torch imports. Detects NVIDIA GPU via nvidia-smi and
installs the matching CUDA wheel if the current torch is CPU-only or
targets the wrong CUDA version.
"""

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from loguru import logger

# PyTorch CUDA wheel tags, ordered newest-first.
# Each entry: (minimum driver CUDA version, wheel tag)
_CUDA_WHEELS = [
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((11, 8), "cu118"),
]

# Env var set after bootstrap installs torch, so the restarted process
# skips the nvidia-smi call and trusts the installed version.
_BOOTSTRAP_ENV = "VRAUTOMATTE_TORCH_OK"


def _nvidia_cuda_version() -> tuple[int, int] | None:
    """Return the driver CUDA version from nvidia-smi, or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", r.stdout)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _best_wheel(driver_cuda: tuple[int, int]) -> str | None:
    """Pick the highest PyTorch CUDA wheel compatible with the driver."""
    for min_ver, tag in _CUDA_WHEELS:
        if driver_cuda >= min_ver:
            return tag
    return None


def _installed_torch_tag() -> str | None:
    """Return the local-version tag of installed torch ('cu128', 'cpu', ...)."""
    try:
        v = pkg_version("torch")  # e.g. "2.10.0+cu128" or "2.10.0"
        return v.split("+")[1] if "+" in v else "cpu"
    except PackageNotFoundError:
        return None


def _install_torch(wheel_tag: str) -> bool:
    """Install torch + torchvision from the PyTorch wheel index."""
    index = f"https://download.pytorch.org/whl/{wheel_tag}"
    candidates = [
        ["uv", "pip", "install", "torch", "torchvision",
         "--index-url", index, "--force-reinstall", "--quiet"],
        [sys.executable, "-m", "pip", "install", "torch", "torchvision",
         "--index-url", index, "--force-reinstall", "--quiet"],
    ]
    for cmd in candidates:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _restart() -> None:
    """Re-launch the current process so the new torch build is loaded."""
    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = "1"

    # Use the module entry point — works regardless of how the app was invoked
    args = [sys.executable, "-m", "vrautomatte.main"] + sys.argv[1:]
    sys.exit(subprocess.call(args, env=env))


def ensure_correct_torch() -> None:
    """Detect GPU and ensure a compatible PyTorch is installed.

    Re-launches the current process if PyTorch was reinstalled so the
    new build is loaded cleanly.
    """
    # Skip if we already verified in a parent invocation
    if os.environ.get(_BOOTSTRAP_ENV):
        os.environ.pop(_BOOTSTRAP_ENV, None)
        return

    driver_cuda = _nvidia_cuda_version()

    if driver_cuda is None:
        return  # No NVIDIA GPU — CPU/MPS torch from PyPI is fine

    target = _best_wheel(driver_cuda)
    if target is None:
        logger.warning(
            f"NVIDIA driver CUDA {driver_cuda[0]}.{driver_cuda[1]} "
            "is too old for available PyTorch CUDA wheels"
        )
        return

    installed = _installed_torch_tag()

    if installed == target:
        return  # Correct variant already installed

    if installed == "cpu" or installed is None:
        logger.warning(
            f"CPU-only PyTorch detected but NVIDIA GPU available "
            f"(CUDA {driver_cuda[0]}.{driver_cuda[1]})"
        )
    else:
        logger.warning(
            f"PyTorch {installed} installed but {target} needed "
            f"for CUDA {driver_cuda[0]}.{driver_cuda[1]}"
        )

    logger.info(
        f"Installing PyTorch with {target} support — this may take a minute..."
    )

    if _install_torch(target):
        logger.info(f"PyTorch {target} installed, restarting...")
        _restart()
    else:
        logger.error(
            "Failed to install CUDA PyTorch. Continuing with current version."
        )
