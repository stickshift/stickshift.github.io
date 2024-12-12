---
title: "Transformer Teardown: Build Your Own Llama Development Kit"
subtitle: "Walk Away with a Collection of Reusable Building Blocks You Can Mix and Match in Your Own Research"
published: "2024-12-11"
banner: resources/banner.png
bibliography:
  - articles/llama-12344/references.bib
kernelspec:
  name: llama-12344
---

```{code-cell} python
:tags: remove-cell
import json
from pathlib import Path
from typing import Callable, Iterator, Mapping, NamedTuple, Sequence, override

from llama_models.llama3.api import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def default_arg[T](
    v: T,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
):
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return default
```

In our last post, we dissected the Llama 3 language model from Meta. We walked through each stage of the pipeline one line of code at a time, getting a close-up view of the machinery powering a state-of-the-art generative Transformer.

The goal of this post is to use what we learned to create a lightweight Llama development kit we can use to run our own experiments. But wait, why can't I just use `transformers` or `ollama`? You certainly could. There are plenty of open source Llama implementations out there. The problem is they all have baggage. They're over complicated with configuration switches and extra options to the point that the main ideas are completely obscured. Not only does this make it hard to understand what's happening, it makes it even harder to run experiments.

:::{card}
*Building your own lightweight stack of Llama components will do wonders for your Transformer fundamentals and you'll walk away with a collection of reusable building blocks you can mix and match in your own research.*
:::

# Pipeline Review

Let's quickly review the stages of a text generation pipeline we learned about in the previous post. The `Tokenize` stage splits raw text into tokens. The `Embeddings` stage maps token ids to token embeddings. The `Context Layers` stage transforms token embeddings into semantic embeddings through layers of attention and feedforward networks. The `Head` stage uses the semantic embeddings as features to predict the next token in the sequence. The predicted token is fed back to the beginning of the pipeline and the process repeats.

```{figure} resources/transformer-pipeline.svg
:label: transformer-pipeline-fig

Transformer Pipeline
```

# Components

In the last post, we broke the pipeline in {ref}`transformer-pipeline-fig` into tiny pieces. In this post, we'll reassemble the pieces into a collection of reusable PyTorch modules shown in {ref}`modules-fig`. `Tokenizer` translates between raw text and token ids. `Generator` is implemented in terms of `Model` and `Head` submodules. `Model` combines an `Embeddings` module with multiple `Layer`s. Each `Layer` is broken into `Attention` and `FFN` submodules. Once `Model` has transformed token embeddings to semantic embeddings, `Head` predicts the next token id. Finally, `Generator` implements the autoregressive decoding loop, feeding the predicted tokens back to `Model`.

```{figure} resources/modules.svg
:label: modules-fig

Modules
```

{ref}`next-token-sequence-fig` illustrates the interactions between each component in the process of predicting the next token.

```{figure} resources/next-token-sequence.svg
:label: next-token-sequence-fig

Predicting the Next Token
```

Over the rest of this post we'll implement each of these components along with a few utilities for loading model configuration and parameters. If you want to jump right to the end, you can find a complete implementation of the [Llama Development Kit](https://github.com/stickshift/llama-kit) on GitHub.

# Model Config

Meta has published multiple versions and configurations of Llama. Each flavor of Llama is represented by a *checkpoint* that includes a model configuration file (`params.json`) and the model parameters (`consolidated.00.pth`).

:::{note}
Before you get started, you'll need to download Llama checkpoints from [www.llama.com](https://www.llama.com/).
:::

Here, we define a `ModelConfig` data structure and `load_config` function to load the checkpoint-specific hyperparameters from `params.json`. Examples include the number of parameters in each layer (`d_model`), the number of layers (`n_layers`), and the number of attention heads (`n_heads`). We'll come back to the model parameters at the end.

```{code-cell} python
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
```

```{code-cell} python
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
```

# Embeddings

Next, we'll implement a `LlamaEmbeddings` module that maps token ids to token embeddings.

```{code-cell} python
class LlamaEmbeddings(nn.Embedding):
    """Llama token embeddings layer."""

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=device,
        )
```

# Context Layers

Next, we'll implement `LlamaAttention`, `LlamaFFN`, and `LlamaLayer` modules that implement the attention and feedforward network blocks in a single decoder layer. This where all the Transformer magic happens, and there is a lot going on here. For the purposes of this post, the important takeaway is simply that the logic is arranged into reusable building blocks.

:::{card}
*For more details on the implementation, please refer to [Transformer Teardown: Llama 3.1](https://stickshift.github.io/articles/llama1/) where we walk through each step one by one.*
:::

## Rotary Position Embedding (RoPE)

```{code-cell} python
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
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head  # noqa: PT018
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head  # noqa: PT018

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
```

## Attention

```{code-cell} python
class LlamaAttention(nn.Module):
    """Llama attention layer."""

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.config = config

        # Input normalization
        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        # Queries projection
        self.w_queries = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_heads * config.d_head,
            bias=False,
            device=device,
        )

        # Keys projection
        self.w_keys = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )

        # Values projection
        self.w_values = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )

        # Output projection
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
        mask = torch.ones(n, n, dtype=torch.bool, device=device) \
                    .tril(diagonal=0)
        m = torch.zeros(n, n, device=device) \
                 .masked_fill_(mask.logical_not(), float("-inf"))

        # Compute attention for all heads in parallel
        scores = q @ k.transpose(-2, -1) / np.sqrt(self.config.d_head) + m
        a = F.softmax(scores, dim=-1) @ v

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
```

## FFN

```{code-cell} python
class LlamaFFN(nn.Module):
    """Llama feed-forward network."""

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        # Input normalization
        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        # Input projection
        self.w_input = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=device,
        )

        # Gate projection
        self.w_gate = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=device,
        )

        # Output projection
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
```

## Layer

```{code-cell} python
class LlamaLayer(nn.Module):
    """Llama transformer layer."""

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
```

# Model

Next, we'll implement a `LlamaModel` component that bundles the embeddings and context layers together. This `model = embeddings + context layers` design is a common pattern in Hugging Face and other Transformer implementations.

```{code-cell} python
class LlamaModel(nn.Module):
    """Combines embeddings and layers in reusable module."""

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.config = config

        self.embeddings = LlamaEmbeddings(config, device)

        self.layers = nn.ModuleList(
            LlamaLayer(config, device) for _ in range(config.n_layers)
        )

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
```

# Head

Next, we'll implement `LlamaHead` and `LlamaCausalLMHead` modules. `LlamaHead` is an abstract module that serves as a base class for task-specific head layers. `LlamaHead` provides common functions such as loading checkpoint weights and projecting the semantic embeddings back to token space. `LlamaCausalLMHead` extends `LlamaHead` to implement "causal language modeling" (aka. next token prediction) based on `temperature`, `top_k`, and `top_p` token sampling.

```{code-cell} python
class LlamaHead(nn.Module):
    """Llama prediction head."""

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        # Input normalization
        self.normalize = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        ).to(device)

        # Output projection
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
    """Llama causal language model head."""

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

        # Temperature
        # -----------

        # If temperature is 0, return the top token
        if self.temperature == 0:
            return torch.argmax(x, dim=-1).item()

        # Apply temperature
        x = x / self.temperature

        # Ranking
        # -------

        # Convert logits to probabilities
        probs = F.softmax(x, dim=-1)

        # Sort probabilities in descending order
        probs, indices = probs.sort(descending=True)

        # Top K
        # -----

        # Retain top k tokens
        probs = probs[: self.top_k]

        # Top P
        # -----

        # Find cutoff where cumulative probability exceeds top_p
        cumulative_mask = probs.cumsum(dim=-1) > self.top_p
        threshold_index = torch.argmax(cumulative_mask).item()

        # Only apply threshold if top_p was exceeded
        if cumulative_mask.any():
            probs = probs[: threshold_index + 1]

        # Random Selection
        # ----------------

        # Sample from remaining tokens weighted by probability
        sampled_index = torch.multinomial(probs, 1)

        # Convert sampled_index to original logits
        token_id = indices[sampled_index]

        return token_id.item()
```

# Generator

Next, we'll implement the `LlamaGenerator` module that combines `LlamaModel` with `LlamaCausalLMHead`. Given a sequence of token ids, `LlamaGenerator` starts by predicting the next token in the sequence before feeding the predicted token back into the model in an autoregressive decoding loop. `LlamaGenerator` continues generating new tokens until it predicts a stop token or exceeds the `max_tokens` parameter.

:::{note}
This is not a production-grade implementation. I've intentionally left out standard optimizations such as query and key caching for the sake of keeping the logic cleaner and easier to understand.
:::

```{code-cell} python
class LlamaGenerator(nn.Module):
    """Llama text generator."""

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
        """Generate token ids until stop token or we exceed max tokens."""
        
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
```

# Model Parameters

# Tokenizer

# Pipeline

# Humpy Dumpty



