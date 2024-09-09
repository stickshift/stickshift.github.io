"""
Minimalistic DistilBERT utilities based on Hugging Face's transformers.models.distilbert.
"""

from pydantic import BaseModel, Field

__all__ = [
    "Config",
    "config",
]


class Config(BaseModel):
    vocab_size: int
    d_model: int
    batch_size: int = Field(default=1)


def config(model) -> Config:
    """Load config from llama model."""

    config = model.config

    return Config(**{
        "vocab_size": config.vocab_size,
        "d_model": config.hidden_size,
    })
