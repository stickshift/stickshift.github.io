import pytest
import torch

import stickshift as ss

__all__ = [
    "device",
]


@pytest.fixture
def device() -> torch.device:
    return ss.torch.device()
