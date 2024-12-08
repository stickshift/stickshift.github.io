from pathlib import Path

import dotenv
import pytest
from pytest import Config

__all__ = [
    "kernel_name",
    "workspace_env",
    "workspace_path",
]


@pytest.fixture
def workspace_path(pytestconfig: Config) -> Path:
    return pytestconfig.rootpath


@pytest.fixture(autouse=True)
def workspace_env(workspace_path: Path):
    env_path = workspace_path / ".env"
    if not env_path.exists():
        raise ValueError(f"Missing .env file in {workspace_path}")

    dotenv.load_dotenv(workspace_path / ".env")


@pytest.fixture
def kernel_name() -> str:
    return "workspace"
