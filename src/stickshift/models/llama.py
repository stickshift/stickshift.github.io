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
    batch_size: int = Field(default=1)
    d_model: int
    n_layers: int
    rms_norm_eps: float
    n_heads: int
    d_head: int
    n_kv_heads: int
    n_kv_groups: int
    rope_base: float
    d_fnn: int


def config(model) -> Config:
    """Load config from llama model."""

    config = model.config

    return Config(**{
        "vocab_size": config.vocab_size,
        "d_model": config.hidden_size,
        "n_layers": config.num_hidden_layers,
        "rms_norm_eps": config.rms_norm_eps,
        "n_heads": config.num_attention_heads,
        "d_head": int(config.hidden_size / config.num_attention_heads),
        "n_kv_heads": config.num_key_value_heads,
        "n_kv_groups": config.num_attention_heads / config.num_key_value_heads,
        "rope_base": 500000,
        "d_fnn": config.intermediate_size,
    })
