import hashlib
from itertools import islice
from secrets import token_hex
import subprocess
from typing import Iterable, Any

__all__ = [
    "default_arg",
    "random_string",
    "shell",
    "md5",
    "take",
]


def default_arg(x, default_factory):
    """Shorthand for if x is None: x = default_factory()."""
    if x is None:
        return default_factory()

    return x


def random_string(length: int | None = None) -> str:
    """Generate random string of specified length."""
    # Defaults
    length = default_arg(length, lambda: 8)

    return token_hex(length // 2 + 1)[0:length]


def shell(command: str) -> str:
    """Run shell command."""

    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")

    return result.stdout.strip()


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def take(n: int, iterable: Iterable[Any]):
    """Process items n at a time."""

    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            break
        yield chunk
