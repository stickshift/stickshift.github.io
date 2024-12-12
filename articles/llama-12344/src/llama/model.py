from functools import partial
import json
from pathlib import Path
from typing import Iterator, Mapping, NamedTuple, Protocol, Sequence, override

from llama_models.llama3.api import Tokenizer as LlamaTokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tools import default_arg

__all__ = [
    "Checkpoint",
    "LlamaAttention",
    "LlamaCausalLMHead",
    "LlamaFFN",
    "LlamaGenerator",
    "LlamaHead",
    "LlamaModel",
    "ModelConfig",
    "Tokenizer",
    "generate_text",
    "load_parameters",
    "load_config",
    "load_tokenizer",
]


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


class ModelConfig(NamedTuple):
    """Llama3 model config."""

    checkpoint_path: Path

    vocab_size: int

    d_model: int

    d_head: int

    d_ffn: int

    n_layers: int

    n_heads: int

    n_kv_heads: int

    rms_norm_eps: float

    rope_theta: float


def load_config(checkpoint_name: str, **kwargs) -> ModelConfig:
    """Load Llama3 config from checkpoint params.json."""
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

    data = {
        "checkpoint_path": checkpoint_path,
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
    }

    # Override with kwargs
    data |= kwargs

    return ModelConfig(**data)


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

ModelParameters = Mapping[str, Tensor]
"""Maps parameter names to weights."""


def load_parameters(config: ModelConfig, **kwargs) -> ModelParameters:
    """Load model state from checkpoint."""
    # Load state from checkpoint
    checkpoint_params = torch.load(
        config.checkpoint_path / "consolidated.00.pth",
        weights_only=True,
        **kwargs,
    )

    # Map keys from Meta's layout
    params = {}

    # Embeddings
    params |= {
        "model.embeddings.weight": checkpoint_params["tok_embeddings.weight"],
    }

    # Layers
    for layer_id in range(config.n_layers):
        params |= {
            f"model.layers.{layer_id}.attention.normalize.weight": checkpoint_params[f"layers.{layer_id}.attention_norm.weight"],
            f"model.layers.{layer_id}.attention.w_queries.weight": checkpoint_params[f"layers.{layer_id}.attention.wq.weight"],
            f"model.layers.{layer_id}.attention.w_keys.weight": checkpoint_params[f"layers.{layer_id}.attention.wk.weight"],
            f"model.layers.{layer_id}.attention.w_values.weight": checkpoint_params[f"layers.{layer_id}.attention.wv.weight"],
            f"model.layers.{layer_id}.attention.w_output.weight": checkpoint_params[f"layers.{layer_id}.attention.wo.weight"],
            f"model.layers.{layer_id}.ffn.normalize.weight": checkpoint_params[f"layers.{layer_id}.ffn_norm.weight"],
            f"model.layers.{layer_id}.ffn.w_input.weight": checkpoint_params[f"layers.{layer_id}.feed_forward.w3.weight"],
            f"model.layers.{layer_id}.ffn.w_gate.weight": checkpoint_params[f"layers.{layer_id}.feed_forward.w1.weight"],
            f"model.layers.{layer_id}.ffn.w_output.weight": checkpoint_params[f"layers.{layer_id}.feed_forward.w2.weight"],
        }

    # Head
    params |= {
        "head.normalize.weight": checkpoint_params["norm.weight"],
        "head.w_output.weight": checkpoint_params["output.weight"],
    }

    return params


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------


class Encoder(Protocol):
    def __call__(self, text: str) -> Sequence[int]: ...


class Decoder(Protocol):
    def __call__(self, token_ids: Sequence[int]) -> str: ...


class Tokenizer(NamedTuple):
    encode: Encoder

    decode: Decoder

    stop_tokens: Sequence[int]


def load_tokenizer(config: ModelConfig) -> Tokenizer:
    """Load tokenizer from checkpoint."""
    # Load tiktoken model
    tokenizer = LlamaTokenizer(str(config.checkpoint_path / "tokenizer.model"))

    # Configure encode parameters
    encode = partial(tokenizer.encode, bos=True, eos=False, allowed_special="all")

    # Configure decode parameters
    decode = tokenizer.decode

    return Tokenizer(
        encode=encode,
        decode=decode,
        stop_tokens=tokenizer.stop_tokens,
    )


# ------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------


def rope_frequencies(config: ModelConfig, device: torch.device, n: int):
    """Compute RoPE cos and sin rotation matrices."""
    # Hyperparameters
    base = config.rope_theta
    d = config.d_head

    # Calculate thetas
    i = torch.arange(d // 2, device=device)
    thetas = base ** (-2 * i / d)

    # Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    thetas = thetas.repeat_interleave(2)

    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = torch.stack([m * thetas for m in range(n)])

    # Apply cos, sin
    r_cos = torch.cos(theta_stack)
    r_sin = torch.sin(theta_stack)

    # Sanity check
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head

    return r_cos, r_sin


def rope_swap(x):
    """Maps [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    # Preserve original shape
    s = x.shape

    # Split into pairs, swap, and restore shape
    x = x.reshape(-1, 2).flip(-1).view(s)

    # Multiply every even index along the last dimension by -1
    #   e.g. [x0, x1, x2, x3] -> [-x0, x1, -x2, x3]
    x[..., ::2] *= -1

    return x


def rope_rotate(x, r_cos, r_sin):
    """Rotate embeddings using RoPE transform."""
    return (x * r_cos) + (rope_swap(x) * r_sin)


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.config = config

        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        self.w_queries = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_heads * config.d_head,
            bias=False,
            device=device,
        )

        self.w_keys = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )

        self.w_values = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )

        self.w_output = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
            device=device,
        )

    @override
    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor) -> Tensor:
        # Match input device
        device = x.device

        # Save residuals
        residual = x

        # Normalize inputs
        x = self.normalize(x)

        # Project inputs to query, key, value spaces
        q = self.w_queries(x)
        k = self.w_keys(x)
        v = self.w_values(x)

        # Split attention heads
        q = self._split_heads(q, self.config.n_heads)
        k = self._split_heads(k, self.config.n_kv_heads)
        v = self._split_heads(v, self.config.n_kv_heads)

        # Expand key/value groups
        reps = self.config.n_heads // self.config.n_kv_heads
        k = k.repeat_interleave(reps, dim=0)
        v = v.repeat_interleave(reps, dim=0)

        # Encode positions by rotating queries and keys
        q = rope_rotate(q, r_cos, r_sin)
        k = rope_rotate(k, r_cos, r_sin)

        # Compute masked attention bias M
        n = len(x)
        mask = torch.ones(n, n, dtype=torch.bool, device=device).tril(diagonal=0)
        m = torch.zeros(n, n, device=device).masked_fill_(mask.logical_not(), float("-inf"))

        # Compute attention for all heads in parallel
        a = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.config.d_head) + m, dim=-1) @ v

        # Combine attention heads
        a = self._combine_heads(a)

        # Project outputs back to model space
        a = self.w_output(a)

        # Merge outputs with residuals
        x = residual + a

        return x

    def _split_heads(self, x: Tensor, n_heads: int):
        """Split attention heads."""
        return x.view(-1, n_heads, self.config.d_head).transpose(-3, -2)

    def _combine_heads(self, x):
        """Combine attention heads."""
        return x.transpose(-3, -2).contiguous().view(-1, int(self.config.n_heads * self.config.d_head))


# ------------------------------------------------------------------------------
# FFN
# ------------------------------------------------------------------------------


class LlamaFFN(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        self.w_input = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=device,
        )

        self.w_gate = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=device,
        )

        self.w_output = nn.Linear(
            in_features=config.d_ffn,
            out_features=config.d_model,
            bias=False,
            device=device,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Save residuals
        residual = x

        # Normalize inputs
        x = self.normalize(x)

        # Apply SwiGLU transform
        f = F.silu(self.w_gate(x)) * self.w_input(x)

        # Project outputs back to model space
        f = self.w_output(f)

        # Merge outputs with residuals
        x = residual + f

        return x


# ------------------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------------------


class LlamaLayer(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.attention = LlamaAttention(config, device)

        self.ffn = LlamaFFN(config, device)

    @override
    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor) -> Tensor:
        # Attention
        x = self.attention(x, r_cos, r_sin)

        # FFN
        x = self.ffn(x)

        return x


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=device,
        )

        self.layers = nn.ModuleList(LlamaLayer(config, device) for _ in range(config.n_layers))

    @override
    def forward(self, token_ids: Tensor) -> Tensor:
        # Match input device
        device = token_ids.device

        # Compute cos and sin rotation matrices once for entire sequence
        r_cos, r_sin = rope_frequencies(self.config, device, len(token_ids))

        # Map tokens to embeddings
        x = self.embeddings(token_ids)

        # Transform token embeddings to semantic embeddings
        for layer in self.layers:
            x = layer(x, r_cos, r_sin)

        return x


# ------------------------------------------------------------------------------
# Head
# ------------------------------------------------------------------------------


class LlamaHead(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        self.w_output = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
            device=device,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Normalize inputs
        x = self.normalize(x)

        # Use last embedding to represent the entire sequence
        x = x[-1]

        # Project outputs to token space
        x = self.w_output(x)

        return x


class LlamaCausalLMHead(LlamaHead):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        super().__init__(config, device)

        self.temperature = default_arg(temperature, 0.6)
        self.top_k = default_arg(top_k, 50)
        self.top_p = default_arg(top_p, 0.9)

    @override
    def forward(self, x: Tensor) -> int:
        # Project semantic embeddings to token space
        x = super().forward(x)

        # If temperature is 0, return the top token
        if self.temperature == 0:
            return torch.argmax(x, dim=-1).item()

        # ---------------------------------------------------------------------
        # Temperature
        # ---------------------------------------------------------------------

        # Apply temperature
        x = x / self.temperature

        # ---------------------------------------------------------------------
        # Ranking
        # ---------------------------------------------------------------------

        # Convert logits to probabilities
        probs = F.softmax(x, dim=-1)

        # Sort probabilities in descending order
        probs, indices = probs.sort(descending=True)

        # ---------------------------------------------------------------------
        # Top K
        # ---------------------------------------------------------------------

        # Retain top k tokens
        probs = probs[: self.top_k]

        # ---------------------------------------------------------------------
        # Top P
        # ---------------------------------------------------------------------

        # Find cutoff where cumulative probability exceeds top_p
        cumulative_mask = probs.cumsum(dim=-1) > self.top_p
        threshold_index = torch.argmax(cumulative_mask).item()

        # Only apply threshold if top_p was exceeded
        if cumulative_mask.any():
            probs = probs[: threshold_index + 1]

        # ---------------------------------------------------------------------
        # Random Selection
        # ---------------------------------------------------------------------

        # Sample from remaining tokens weighted by probability
        sampled_index = torch.multinomial(probs, 1)

        # Convert sampled_index to original logits
        token_id = indices[sampled_index]

        return token_id.item()


# ------------------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------------------


class LlamaGenerator(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ):
        super().__init__()

        self.device = device

        self.stop_tokens = load_tokenizer(config).stop_tokens

        self.max_tokens = default_arg(max_tokens, 32)

        self.model = LlamaModel(config, device)

        self.head = LlamaCausalLMHead(
            config,
            device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def __call__(self, token_ids: Sequence[int]) -> Iterator[int]:
        # Prepare model
        self.model.eval()
        self.head.eval()

        # Make mutable copy of token ids
        token_ids = list(token_ids)

        with torch.no_grad():
            # Generate output until we get a stop token or we exceed max_tokens.
            for _ in range(self.max_tokens):
                # Load token ids into a tensor
                x = torch.tensor(token_ids, device=self.device)

                # Transform token_ids into semantic embeddings
                x = self.model(x)

                # Predict next token
                token_id = self.head(x)

                # Check stopping criteria
                if token_id in self.stop_tokens:
                    break

                # Yield token
                yield token_id

                # Append to end of sequence
                token_ids.append(token_id)
