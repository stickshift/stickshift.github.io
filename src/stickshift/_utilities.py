from secrets import token_hex

__all__ = [
    "default_arg",
    "random_string",
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
