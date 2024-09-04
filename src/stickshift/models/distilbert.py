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
    n_heads: int
    d_head: int
    d_fnn: int
    n_layers: int
    n_labels: int
    max_sequence_length: int
    batch_size: int = Field(default=1)


def config(model) -> Config:
    """Load config from distilbert model."""

    config = model.distilbert.config

    return Config(**{
        "vocab_size": config.vocab_size,
        "d_model": config.dim,
        "n_heads": config.n_heads,
        "d_head": int(config.dim / config.n_heads),
        "d_fnn": config.hidden_dim,
        "n_layers": config.n_layers,
        "n_labels": config.num_labels,
        "max_sequence_length": config.max_position_embeddings,
    })
