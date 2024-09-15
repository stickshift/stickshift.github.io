import pytest
import torch

from stickshift.torch import device as torch_device

__all__ = [
    "device",
]


@pytest.fixture
def device() -> torch.device:
    return torch_device()
