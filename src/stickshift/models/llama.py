"""Minimalistic Llama3 utilities based on Meta's reference implementation."""

import json
from pathlib import Path

from pydantic import BaseModel
import torch

from stickshift import default_arg, take

__all__ = [
    "Config",
    "load_checkpoint",
    "load_state",
    "rotate_half",
]


class Config(BaseModel):
    """Custom Llama3 config."""

    checkpoint_path: Path
    vocab_size: int
    d_model: int
    d_head: int
    d_ffn: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    n_kv_groups: int
    rms_norm_eps: float
    rope_theta: float


def load_checkpoint(
    checkpoint_name: str,
    max_seq_len: int | None = None,
    device: torch.device | None = None,
) -> tuple[Config, dict]:
    """Load Llama 3 checkpoint."""
    # Defaults
    max_seq_len = default_arg(max_seq_len, lambda: 8192)

    # Build checkpoint_path
    checkpoints_path = Path("~/.llama/checkpoints").expanduser()
    checkpoint_path = checkpoints_path / checkpoint_name

    # Load hyperparameters
    hparams_path = checkpoint_path / "params.json"
    hparams = json.loads(hparams_path.read_text())

    # Calculate d_ffn from 8/3 * d_model rounded to nearest multiple_of
    d_model = hparams["dim"]
    ffn_dim_multiplier = hparams["ffn_dim_multiplier"]
    multiple_of = hparams["multiple_of"]
    d_ffn = int(8 / 3 * d_model * ffn_dim_multiplier)
    d_ffn = multiple_of * ((d_ffn + multiple_of - 1) // multiple_of)

    config = Config(**{
        "checkpoint_path": checkpoint_path,
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "n_kv_groups": hparams["n_heads"] / hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
    })

    # Load model params
    checkpoint = torch.load(checkpoint_path / "consolidated.00.pth", weights_only=True, map_location=device)

    return config, checkpoint


def load_state(*args, checkpoint, layer=None):
    # Defaults
    layer = default_arg(layer, lambda: 0)

    for module, key in take(2, args):
        match key:
            case "embeddings":
                module.load_state_dict({
                    "weight": checkpoint["tok_embeddings.weight"],
                })
            case "normalize_attention":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention_norm.weight"],
                })
            case "normalize_ffn":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.ffn_norm.weight"],
                })
            case "normalize_head":
                module.load_state_dict({
                    "weight": checkpoint["norm.weight"],
                })
            case "w_q":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wq.weight"],
                })
            case "w_k":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wk.weight"],
                })
            case "w_v":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wv.weight"],
                })
            case "attention_outputs":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wo.weight"],
                })
            case "ffn_gates":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w1.weight"],
                })
            case "ffn_inputs":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w3.weight"],
                })
            case "ffn_outputs":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w2.weight"],
                })
            case "head_outputs":
                module.load_state_dict({
                    "weight": checkpoint["output.weight"],
                })
            case _:
                raise ValueError(f"Unexpected key {key}")


def rotate_half(x):
    """Convert x to -x1, x0, ..."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
