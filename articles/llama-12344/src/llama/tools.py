import subprocess
from typing import Callable

import torch

__all__ = [
    "default_arg",
    "shell",
    "torch_device",
]


def default_arg[T](
    v: T,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
):
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return default


def shell(command: str) -> str:
    """Run shell command."""
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def torch_device() -> torch.device:
    """Configure gpus."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
