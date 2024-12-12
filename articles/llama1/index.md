---
title: "Transformer Teardown: Llama 3.1"
subtitle: "Trace an Inference Through Each Layer of the SOTA Llama 3.1 Foundation Models"
published: "2024-09-21"
banner: resources/banner.png
bibliography:
  - articles/llama1/references.bib
kernelspec:
  name: llama1
---

In [the last Transformer Teardown](https://stickshift.github.io/articles/distilbert1), we dissected a DistilBERT text classification pipeline, tracing a single inference through the entire stack from raw data to final prediction. Studying BERT-based text classification models is a fantastic way to see the basic Transformer machinery in action. But BERT was published in 2018! It would be another 4 years before ChatGPT launched and Generative AI exploded onto the scene. It's safe to say a lot has changed.

In this Transformer Teardown, we're going to fast forward to present day. We'll use the same teardown process to unpack the state-of-the-art [Llama 3.1](https://llama.meta.com/) open source foundation models released by Meta in July. We'll walk through each step of a text generation pipeline one cell at a time, tracing an inference from raw text to the first output token. We'll illustrate the main ideas from the latest Transformer literature with minimal, straightforward, working Python code, giving you a close-up view of the core mechanisms driving the Generative AI revolution.

```{code-cell} ipython3
---
tags: [remove-cell]
---
#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from functools import partial
from itertools import islice
import json
import math
import os
from pathlib import Path
from sys import stdout
from textwrap import dedent
from typing import Any, Iterable, NamedTuple, Sequence
import warnings

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pandas import Series
from pytest import approx
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.nn.functional import relu, silu, softmax

from llama_models.llama3.reference_impl.model import RMSNorm
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
#-------------------------------------------------------------------------------
# General Utilities
#-------------------------------------------------------------------------------

def default_arg(x, default=None, default_factory=None):
    if x is None:
        if default_factory is not None:
            return default_factory()
        
        return default

    return x


def torch_device() -> torch.device:
    """Configure gpus."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
    

def take(n: int, iterable: Iterable[Any]):
    """Process items n at a time."""
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            break
        yield chunk
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
#-------------------------------------------------------------------------------
# Llama Utilities
#-------------------------------------------------------------------------------

class Config(NamedTuple):
    """Custom Llama3 config."""

    device: torch.device

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

    max_seq_len: int

    temperature: float | None = 0.6

    top_k: int = 50

    top_p: float = 0.9

    max_completion_tokens: int = 64


def load_config(
    checkpoint_name: str,
    device: torch.device | None = None,
    max_seq_len: int | None = None,
    **kwargs,
) -> Config:
    """Load Llama3 config from checkpoint."""
    # Defaults
    device = default_arg(device, default_factory=torch_device)
    max_seq_len = default_arg(max_seq_len, 8192)

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

    defaults = {
        "device": device,
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
        "max_seq_len": max_seq_len,
    }

    # Override with kwargs
    data = defaults | kwargs

    return Config(**data)


def load_state(*args, checkpoint, layer=None):
    # Defaults
    layer = default_arg(layer, 0)

    for module, key in take(2, args):
        match key:
            # Embeddings
            case "embeddings":
                module.load_state_dict({
                    "weight": checkpoint["tok_embeddings.weight"],
                })

            # Attention
            case "normalize_attention":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention_norm.weight"],
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
            case "w_a":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wo.weight"],
                })

            # FFN
            case "normalize_ffn":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.ffn_norm.weight"],
                })
            case "w_g":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w1.weight"],
                })
            case "w_h":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w3.weight"],
                })
            case "w_f":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w2.weight"],
                })

            # Head
            case "normalize_head":
                module.load_state_dict({
                    "weight": checkpoint["norm.weight"],
                })
            case "w_head":
                module.load_state_dict({
                    "weight": checkpoint["output.weight"],
                })
            case _:
                raise ValueError(f"Unexpected key {key}")


def load_pretrained_state(layer):    
    # Load pre-trained state
    load_state(
        normalize_attention, "normalize_attention", 
        w_q, "w_q", 
        w_k, "w_k", 
        w_v, "w_v", 
        w_a, "w_a",
        normalize_ffn, "normalize_ffn", 
        w_g, "w_g",
        w_h, "w_h",
        w_f, "w_f",
        checkpoint=checkpoint,
        layer=layer,
    ) 


def rotate_half(x):
    """Convert x to -x1, x0, ..."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
#-------------------------------------------------------------------------------
# Configure
#-------------------------------------------------------------------------------

# Ignore all warnings
warnings.filterwarnings("ignore")

# Configure gpu
device = torch_device()

# Torch display options
torch.set_printoptions(linewidth=120)

# Disable pretty printing
%pprint
```

# Llama Foundation Models

[Llama](https://llama.meta.com/) is a family of general purpose, state-of-the-art, open source foundation models from Meta. According to the 3.1 technical report, the latest models can "answer questions in at least 8 languages, write high quality code, solve complex reasoning problems, and use tools in a zero-shot way." {cite:t}`dubey_llama_2024`

The Llama 3.1 release includes 8B, 70B, and 405B parameter sizes. While you need a multi-GPU cluster to run the 70B and 405B sizes, the 8B model is small enough to experiment with on a laptop. Not only did Meta release the pre-trained model checkpoints for all 3 sizes, they also published a fantastically detailed, [70 page technical report](https://arxiv.org/abs/2407.21783v2) as well as a complete [reference implementation](https://github.com/meta-llama/llama-models).

:::{card}
*Llama 3.1 represents an incredible learning opportunity to study the inner workings of a modern frontier model.*
:::

Over the course of this post, we'll implement a complete text generation pipeline using only the research literature, pre-trained weights from the `Meta-Llama3.1-8B-Instruct` checkpoint, and Meta's reference implementation as a guide. After we load the 8B checkpoint, we'll review the stages of an end-to-end, text generation pipeline. In the sections that follow, we'll walk through a detailed teardown of each stage—tracing an inference from raw data to the first output token. In the last section, we'll put all the pieces together into a complete generative Transformer capable of producing long form content.

Let the teardown begin!

# Model Checkpoint

We'll start by loading the configuration and pre-trained weights for the `Meta-Llama3.1-8B-Instruct` checkpoint. The "instruct" versions of the Llama models include the raw pre-training and substantial post-training to support user and assistant interactions and complex tool-calling scenarios. The weights for all Llama checkpoints can be downloaded directly from [Meta](https://llama.meta.com/), [Hugging Face](https://huggingface.co/meta-llama), and [Kaggle](https://www.kaggle.com/organizations/metaresearch/models).

```{code-cell} ipython3
# Load model config
config = load_config("Meta-Llama3.1-8B-Instruct")

# Load pre-trained model parameters
checkpoint = torch.load(
    config.checkpoint_path / "consolidated.00.pth", 
    weights_only=True, 
    map_location=device,
)

config._asdict()
```

We'll reference a number of the settings in `config` throughout the teardown. For now, a few interesting ones to note are `d_model`, `d_ffn`, `n_layers`, and `n_heads`. These represent the primary differences between the 8B, 70B, and 405B sizes.


# Text Generation Pipeline

In [the last teardown](/articles/distilbert1/), we looked at a text classification Transformer. This time we're going to dissect a *text generation* Transformer. Instead of simply applying a label to the input text, the Head stage will be responsible for *generating* new content. But don't worry! It's not as complicated as it sounds.

{ref}`text-generation-pipeline-fig` illustrates the stages in a text generation pipeline. It's very similar to the text classification pipeline we looked at last time. The Tokenize stage splits raw text into a sequence of tokens. The Embeddings stage maps the sequence of tokens to a sequence of embedding vectors. The Context Layers augment the embeddings with contextual signals drawn from the surrounding tokens, transforming individual token embeddings into contextualized "semantic embeddings". Finally, the Head stage converts the semantic embeddings into predictions. The key difference is, instead of predicting a label for the raw text, text generation Transformers *predict the next token*.

```{figure} resources/transformer-pipeline.svg
:label: text-generation-pipeline-fig

Text Generation Pipeline
```

But one token is just the beginning! The magical powers of Generative AI are manifested by simply running the token predictions in a loop. The predicted token in each iteration is appended to the end of the input sequence, and the process repeats. Over and over again.


# Raw Text

Before we can tear anything down, we need a prompt. Since our goal is to trace an inference from raw text to the first output token, we want to start with a prompt that's specific enough to generate a consistent, one-word answer. If we do everything right, the first output token we predict should be "Boston".

```{code-cell} ipython3
# Prompt
prompt = "<|start_header_id|>user<|end_header_id|>\n\n"
prompt += "What is the capital of Massachusetts? Answer in one word."
prompt += "<|eot_id|>"
prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

You can see `prompt` includes a number of special tokens. These would usually be injected by a framework like Hugging Face's `transformers`. We need to manually inject them because we're working with the model directly. You can find more information on the Llama 3.1 prompt syntax in the [Llama Prompting Guide](https://www.llama.com/docs/how-to-guides/prompting).


# Tokenize

The Tokenize stage splits raw text into a sequence of tokens using a fixed vocabulary. Llama uses a vocabulary of 128k tokens built on top of OpenAI's tiktoken tokenizer. We'll dig into the gory details in the later stages, but here we'll simply use the off-the-shelf Tokenizer from Meta's llama-models reference implementation.

```{code-cell} ipython3
from llama_models.llama3.api.tokenizer import Tokenizer

# Load tokenizer model from checkpoint
tokenizer = Tokenizer(str(config.checkpoint_path / "tokenizer.model"))
```

```{code-cell} ipython3
# Split raw text into tokens
token_ids = tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")
token_ids
```

```{code-cell} ipython3
len(token_ids)
```

We see `tokenizer.encode` split our prompt into 22 token ids. These ids represent the index of each token in Llama's 128k token vocabulary. We can always reverse the process with `tokenizer.decode`. If you look closely at the cell output below, you'll notice the tokenizer injected another special token `(128000, '<|begin_of_text|>')` to mark the beginning of the sequence.

```{code-cell} ipython3
# Decode token ids back into raw text
tokenizer.decode(token_ids)
```

```{code-cell} ipython3
# Load token_ids into a tensor
x = torch.tensor(token_ids, device=device)

x.shape
```


# Embeddings

Embeddings are a key component of the Transformer architecture. They're also abstract mathematical structures that can be difficult to wrap your head around. To illustrate the crucial role embeddings play, let's use a quick metaphor.

:::{card}
*If a Transformer was a brain, then embeddings would be the electrical signals carrying information through the brain.*
:::

Continuing with the metaphor, the Embeddings stage of the pipeline would be your sensory organs where light rays and air vibrations are translated into electrical impulses. Token embeddings would be the fresh sensory percepts. Semantic embeddings would be the abstract thoughts at the top of the cortical stack. The idea of percepts traveling up the cortical stack is a perfect analogy for token embeddings traveling through the Transformer layers.

Implementing Llama's Embeddings stage is relatively straightforward. We'll use a lookup table with a unique embedding for each of the 128k tokens in the vocabulary. Each embedding is a vector with {math}`d_{model}` elements that were randomly generated and then learned during training. Given a sequence of token ids, the lookup table returns their embeddings as row vectors stacked in an {math}`n \times d_{model}` tensor. 

```{figure} resources/embeddings.svg
:label: embeddings-fig

Learned Token Embeddings
```

```{code-cell} ipython3
# Initialize embeddings lookup table
embeddings = nn.Embedding(
    num_embeddings=config.vocab_size, 
    embedding_dim=config.d_model,
    device=device,
)

# Load pre-trained state
load_state(embeddings, "embeddings", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Map token ids to embeddings
x = embeddings(x)

x.shape
```

We can see from `x.shape` that we successfully mapped the 22 token ids to 22 token embeddings stacked in an {math}`n \times d_{model}` tensor.

```{code-cell} ipython3
# Show sample
x
```

Before we move on, a quick note on terminology. If you've used cloud-based LLM APIs like OpenAI or LangChain, you're likely familiar with the term "embedding model". An *embedding model* is really a combination of a tokenizer and embeddings table. These are often bundled together to give you everything you need to convert raw text into embedding vectors and can be used for a number of things independent of the LLM.

Now that we've converted our raw text into token embeddings, it's time to start transforming!


# Context Layers

Context layers are where the Transformer magic happens. Collectively, the Context Layers are responsible for transforming a sequence of token embeddings into a sequence of semantic embeddings. The mechanism works by passing the embeddings through multiple layers of attention and feedforward blocks. The attention blocks focus on relationships between embeddings, augmenting each one with a weighted combination of the surrounding embeddings. The feedforward blocks capitalize on the extra context, transforming each augmented embedding with the non-linear magic of a fully-connected multilayer perceptron. By stacking multiple layers together, Transformers repeat the pattern of attention and transformation, gradually converting representations of individual words into representations of abstract semantic concepts.

```{figure} resources/transformer-layers.svg
:label: context-layers-fig

Context Layers
```

{ref}`context-layers-fig` illustrates the flow of information through a single layer. Embeddings are first passed to the Attention block. The attention outputs are added to the attention inputs before being passed to the FFN block. Similarly, the FFN outputs are added to the FFN inputs before being passed to the next layer. Adding the inputs and outputs of each block is known as "residual learning" and is critical for providing a stable path for gradient flow during training {cite:p}`he_deep_2015`.


## Decoder-Only Model Architecture

Like most of today's generative models, Llama uses a "decoder-only" model architecture. Instead of using the fully connected self attention we saw in [the DistilBERT teardown](/articles/distilbert1/), the context layers in Llama use *masked self attention*. The "decoder-only" term comes from the "Attention is All You Need" paper, where Vaswani et al. described layers of self attention as "encoder layers" and layers of masked self attention as "decoder layers". While Vaswani et al.'s *Vanilla* Transformer architecture processed inputs and outputs with encoder layers and decoder layers respectively, later researchers showed that by adding more compute you could achieve the same goals using a single stack of decoder layers. For a fascinating discussion of how decoders became the dominant architecture, I highly recommend watching [Hyung Won Chung's guest lecture at Stanford on the Future of AI](https://youtu.be/orDKvo8h71o?si=J2sxhYtL9LCd6IRk) from April of this year.


## Attention

Attention is the signature component in the Transformer architecture. In the 7 years since Vaswani et al. published "Attention is All You Need", researchers have experimented with numerous attention variations of all shapes and sizes. Before we jump into the code, we'll quickly review the fundamental concepts behind attention followed by details on the specific approach chosen by the Llama authors.


### What is Attention?

Given our input embeddings are stacked in an {math}`n \times d_{model}` tensor {math}`\mathbf{X}`, the goal of attention is to map each embedding {math}`\set{\mathbf{x}_m \mid \mathbf{x}_m \in \mathbf{X}}` to an attention representation {math}`\mathbf{a}_m` that includes relevant contextual signals drawn from the rest of the embeddings {math}`\set{\mathbf{x}_n \mid \mathbf{x}_n \in \mathbf{X}, n \neq m}`. 

For example, let's imagine we've mapped the sentence `I love New York` to the sequence of token embeddings {math}`\mathbf{x} = [E_{I}, E_{love}, E_{New}, E_{York}]`. The embedding {math}`\mathbf{x}_2` represents the word "New" *in isolation*. The word "New" can mean a lot of things; many of which have nothing to do with this sentence. Our goal would be to generate an attention representation {math}`\mathbf{a}_2` containing signals from the other embeddings {math}`\set{E_{I}, E_{love}, E_{York}}` that would help us create a *better* version of {math}`\mathbf{x}_2`:

```{math}
\mathbf{x}_{2*} = \mathbf{x}_{2} + \mathbf{a}_{2}
```

Let's assume each embedding contributes "something" to {math}`\mathbf{a}_m`. Even though we can't quantify "something" yet, we can write {math}`\mathbf{a}_m` as an unknown function {math}`f_A` of the two embeddings {math}`\mathbf{x}_m`, {math}`\mathbf{x}_n`:

```{math}
\mathbf{a}_m = \sum_{\mathbf{x}_n \in \mathbf{X}} f_A(\mathbf{x}_m, \mathbf{x}_n)
```

All of the attention variations in the Transformer literature—e.g. Self Attention, Multi-Head Self Attention, Linear Attention, Grouped Query Attention—are different approaches to implement {math}`f_A`. In practice, the authors of a new Transformer model start with Vaswani et al.'s attention definition and then select from the large, à la carte menu of improvements that have been published since, resulting in their own unique variation of attention.


### Attention in Llama 3.1

Defining attention in Llama 3.1 requires a bit of backtracking. Most of the details can be found in Llama 1 {cite:p}`touvron_llama_2023-1` with a few changes in Llama 2 {cite:p}`touvron_llama_2023` and only minor adjustments in Llama 3 {cite:p}`dubey_llama_2024`.

Starting with the standard Masked Self Attention definition from {cite:t}`vaswani_attention_2017`, Llama adopts the following improvements that affect attention:

* Pre-normalization with RMSNorm: improves training stability and inference speed.
* Grouped Query Attention (GQA) with 8 key/value heads: improves inference speed and reduce memory overhead of key-value caches.
* Rotary Position Embedding (RoPE) with {math}`\Theta = 500,000`: relative position encoding improves performance on longer context windows.

We'll start by describing the standard Masked Self Attention and then describe how these improvements modify the final attention calculation in Llama.


### Masked Self Attention

Given {math}`n` input embeddings of length {math}`d_{model}` stacked in an {math}`n \times d_{model}` tensor {math}`\mathbf{X}`, the standard masked self attention algorithm from Vaswani et al. can be expressed using the following equation:

```{math}
\begin{equation}
\mathbf{A} = softmax\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_K}} + \mathbf{M}\right)\mathbf{V} \\
\end{equation}
```

where

* {math}`\mathbf{Q}`, {math}`\mathbf{K}`, and {math}`\mathbf{V}` are all linear projections of {math}`\mathbf{X}`.
* {math}`\mathbf{Q}` is an {math}`n \times d_K` tensor of *queries* that represent selection criteria for surrounding embeddings that would add valuable context to the current representation.
* {math}`\mathbf{K}` is an {math}`n \times d_K` tensor of *keys* that represent characteristics that satisfy the selection criteria in the queries.
* {math}`\mathbf{V}` is an {math}`n \times d_{model}` tensor of *values* that represent the contextual signals to transfer from one embedding to another.
* The {math}`\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_K}}` term is an {math}`n \times n` tensor of "attention weights" that define how much of each embedding's values to include from {math}`\mathbf{V}`.
* {math}`\mathbf{M}` is an {math}`n \times n` attention mask that prevents earlier embeddings from attending to later ones by adding a {math}`-\infty` bias to the later embeddings' scores.
* {math}`\mathbf{A}` is an {math}`n \times d_{model}` tensor of attention representations.


The mechanism starts with the {math}`\mathbf{Q}\mathbf{K}^T` term that calculates the *angular distance* between each query vector {math}`\mathbf{q}_m` and key vector {math}`\mathbf{k}_n` by taking their dot product. The smaller the angle, the closer the vectors, and the better the match between {math}`\mathbf{q}_m` and {math}`\mathbf{k}_n`. The result of the {math}`\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_K}}` term is an {math}`n \times n` tensor of raw scores where row {math}`i` represents query {math}`\mathbf{q}_m` and column {math}`j` represents how well key {math}`\mathbf{k}_n` matches {math}`\mathbf{q}_m`.

The next step is to add the attention mask {math}`\mathbf{M}` to prevent earlier embeddings from attending to later embeddings. Imagine {math}`\mathbf{M}` as an {math}`n \times n` diagonal mask with the upper right half set to {math}`-\infty` and the lower left set to {math}`0`. Illustrated below, you can see that {math}`\mathbf{q}_0` can only attend to {math}`\set{\mathbf{k}_0}`, while {math}`\mathbf{q}_1` can attend to {math}`\set{\mathbf{k}_0, \mathbf{k}_1}`, and {math}`\mathbf{q}_2` can attend to {math}`\set{\mathbf{k}_0, \mathbf{k}_1, \mathbf{k}_2}`.

```{figure} resources/attention-mask.svg
:label: attention-mask-fig

Attention Mask
```

Next, the {math}`softmax` term normalizes the attention weights across the keys before they're applied to {math}`\mathbf{V}`.


### Grouped Query Attention (GQA)

Each new token the model generates is compared against every key and value that came before. As context windows grow larger, the memory used by the key and value caches becomes a serious bottleneck. {cite:p}`ainslie_gqa_2023`

To address this bottleneck, Llama 2 replaced the Multi-Head Attention (MHA) mechanism in Llama 1 with technique called Grouped Query Attention (GQA) {cite:p}`ainslie_gqa_2023`. In MHA, each attention head has it's own set of queries, keys, and values. Earlier models such as PaLM {cite:p}`chowdhery_palm_2022` tried replacing MHA with Multi-Query Attention (MQA) which shares a single set of keys and values across all the attention heads. But MQA didn't perform as well as hoped. GQA was designed as a trade-off between the two extremes of MHA and MQA where keys and values *are* shared across attention heads like MQA, but, instead of one group, Llama 3.1 uses 8 key / value heads.

We'll see how GQA is implemented in the upcoming sections on splitting and combining attention heads.


### Rotary Position Embedding (RoPE)

The relevance of one embedding to another is heavily influenced by the distance between them. This makes the embedding positions in the sequence critically important. However, if we recall our unknown attention function {math}`f_A(\mathbf{x}_m, \mathbf{x}_n)`, you may notice the positions are conspicuously missing. This worked for early Transformers like Vanilla and BERT because they encoded the positions directly into the token embeddings {math}`\mathbf{x}_m`, {math}`\mathbf{x}_n`.

More recent models including Llama have adopted *relative* position encoding schemes that have been shown to perform better especially on much longer sequences. Instead of baking the positions into the token embeddings, the idea is to explicitly add the positions {math}`m`, {math}`n`, to our attention function:

```{math}
f_A(\mathbf{x}_m, \mathbf{x}_n, m, n)
```

To put this into practice, Llama uses an approach known as Rotary Position Embedding (RoPE) from {cite:t}`su_roformer_2021`. As we saw earlier, the attention mechanism relies on the angular distance between query and key vectors as a measure of fitness. RoPE intentionally takes advantage of this, converting distance between embedding positions into angular distance between embedding vectors.

This is straightforward to visualize in 2-dimensions. The following diagram shows 2 embeddings {math}`\mathbf{x}_m` and {math}`\mathbf{x}_n` with positions {math}`m` and {math}`n` respectively. The idea of RoPE is to rotate {math}`\mathbf{x}_m` a distance of {math}`m \theta` and {math}`\mathbf{x}_n` a distance of {math}`n \theta`, directly translating the distance in sequence space {math}`(n - m)` to a distance in vector space {math}`(\measuredangle{n} - \measuredangle{m})`.

```{figure} resources/rope.svg
:label: rope-fig

RoPE Concept in 2D
```

While the 2-dimensional concept is intuitive, implementing RoPE with {math}`d_{model}`-dimensional vectors is a little more complicated. The complete RoPE algorithm involves several steps. To see what's happening, it helps to unpack and walk through them one at a time.

Let's start by rotating a single embedding. Given an embedding {math}`\mathbf{x}` with {math}`d` elements and position {math}`m`, our goal is to rotate {math}`\mathbf{x}` an angular distance of {math}`m \theta`. Unfortunately, there isn't an exact solution for this when {math}`d` > 2. Instead, RoPE approximates the idea by splitting {math}`\mathbf{x}` into pairs {math}`\set{(x_0, x_1), (x_2, x_3), \dots, (x_{d-2}, x_{d-1})}` and then rotating each pair in 2D.

Given {math}`\mathbf{x}`, {math}`d`, {math}`m`, {math}`\theta`, we can rotate {math}`\mathbf{x}` an angular distance of {math}`m \theta` by calculating {math}`\mathbf{R}\mathbf{x}`:

```{math}
\mathbf{R} \mathbf{x} = 
\begin{bmatrix}
cos(m \theta) & -sin(m \theta) & 0 & 0 \\
sin(m \theta) & cos(m \theta) & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots \\
0 & 0 & cos(m \theta) & -sin(m \theta) \\
0 & 0 & sin(m \theta) & cos(m \theta) \\
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix}
=
\begin{bmatrix}
x_0 \cdot cos(m \theta) - x_1 \cdot sin(m \theta) \\
x_0 \cdot sin(m \theta) + x_1 \cdot cos(m \theta) \\
\vdots \\
x_{d-2} \cdot cos(m \theta) - x_{d-1} \cdot sin(m \theta) \\
x_{d-2} \cdot sin(m \theta) + x_{d-1} \cdot cos(m \theta) \\
\end{bmatrix}
```

Given the sparsity of {math}`\mathbf{R}`, we'll use the compact form recommended by {cite:t}`su_roformer_2021`. If you compare the compact form below with the right hand side above, you can see they achieve the same result.

Given {math}`\mathbf{x}`, {math}`d`, {math}`m`, {math}`\theta`:

```{math}
\mathbf{R} \mathbf{x} = 
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix}
\odot
\begin{bmatrix}
cos(m \theta) \\
cos(m \theta) \\
\vdots \\
cos(m \theta) \\
cos(m \theta) \\
\end{bmatrix}
+
\begin{bmatrix}
-x_1 \\
x_0 \\
\vdots \\
-x_{d-1} \\
x_{d-2} \\
\end{bmatrix}
\odot
\begin{bmatrix}
sin(m \theta) \\
sin(m \theta) \\
\vdots \\
sin(m \theta) \\
sin(m \theta) \\
\end{bmatrix}
```

Great. We've succeeded at rotating each of the pairs {math}`\set{(x_0, x_1), \dots, (x_{d-2}, x_{d-1})}` an angular distance {math}`m \theta`. However, RoPE takes position encoding a step further by varying the angular offset {math}`\theta` across the pairs. The actual equation RoPE uses to rotate a single embedding is the following:

Given {math}`\mathbf{x}`, {math}`d`, {math}`m`, {math}`\Theta`:

```{math}
\mathbf{R} \mathbf{x} = 
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix}
\odot
\begin{bmatrix}
cos(m \theta_0) \\
cos(m \theta_0) \\
\vdots \\
cos(m \theta_{d/2-1}) \\
cos(m \theta_{d/2-1}) \\
\end{bmatrix}
+
\begin{bmatrix}
-x_1 \\
x_0 \\
\vdots \\
-x_{d-1} \\
x_{d-2} \\
\end{bmatrix}
\odot
\begin{bmatrix}
sin(m \theta_0) \\
sin(m \theta_0) \\
\vdots \\
sin(m \theta_{d/2-1}) \\
sin(m \theta_{d/2-1}) \\
\end{bmatrix}
```

where

```{math}
\theta_i = \frac{1}{\Theta^{2i/d}}, i \in [0, 1, \dots, d/2-1]
```

Let's visualize {math}`\theta_i` for {math}`i = [0, d_{head}/2)` to get a better sense for what's happening here. The following cell plots {math}`\theta_i` using both the original {math}`\Theta = 10000` from {cite:t}`su_roformer_2021` and the {math}`\Theta = 500000` from Llama 3.1. Surely the Llama authors must have had a reason to change it, right?

```{code-cell} ipython3
i = torch.arange(config.d_head//2)

thetas = 10000 ** (-2*i / config.d_head)
sns.lineplot(x=i, y=thetas, label=f"$\Theta=10k$")

thetas = 500000 ** (-2*i / config.d_head)
sns.lineplot(x=i, y=thetas, label=f"$\Theta=500k$")

plt.xlabel("i")
plt.ylabel("theta_i")
```

I'll admit this baffled me for a while. I could understand the rotation idea. *But why would you go through the trouble of varying {math}`\theta` by what seems like such an arbitrary amount?*

To understand the rationale, we need to step back for a moment and think about the big picture. Why are we using embeddings in the first place? At the end of the day, embeddings are feature vectors. Just like rows in a classical ML master table. Except they literally have thousands of columns. We can't, nor do we want to, know what each one represents. There are simply too many of them. But we do want to support them. Like seeds in a garden, we want to give them nutrient rich soil in which to flourish.

If we rotate all the features by a uniform amount, we're restricting the nutrients in their diet. Some features will be sensitive to distance. The effect of these "short-range" or "local" features may taper off rapidly as the embeddings move apart. Other "long-range" or "global" features will be much less sensitive to distance.

:::{card}
*Varying {math}`\theta` establishes an environment that encourages a diverse population of short, medium, and long-range features.*
:::


### Attention Workflow

Now that we've gone through all of the background, let's rewrite the attention equation one more time with all of the elements we need to calculate.
    
```{math}
\begin{aligned}
\mathbf{A} &= softmax\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V} \\
\end{aligned}
```

expands to

```{math}
\begin{aligned}
\mathbf{A} &= softmax\left(\frac{(\mathbf{R}\mathbf{W}_Q\mathbf{X})(\mathbf{R}\mathbf{W}_K\mathbf{X})^T}{\sqrt{d}} + \mathbf{M}\right)\mathbf{W}_V\mathbf{X} \\
\end{aligned}
```

where

* {math}`\mathbf{W}_Q`, {math}`\mathbf{W}_K`, {math}`\mathbf{W}_V` are the linear projections for queries, keys, and values respectively.
* {math}`\mathbf{R}\mathbf{W}_Q\mathbf{X}`, {math}`\mathbf{R}\mathbf{W}_K\mathbf{X}` encode positions by rotating queries and keys respectively.

The flowchart below enumerates the steps required to calculate {math}`\mathbf{A}`. Yes, there are a lot of steps but each one is tiny. My goal was to break the calculation down into small enough pieces that each one is only a few lines of code.

```{figure} resources/attention-flow.svg
:label: attention-workflow-fig

Attention Workflow
```


### Normalize Inputs

In the original Transformer architecture, the Attention and FFN blocks normalized the outputs of each sub-layer. In contrast, Llama normalizes the *inputs* to each sub-layer to improve training stability. Furthermore, Llama also replaces the standard LayerNorm algorithm with the RMSNorm algorithm designed by {cite:t}`zhang_root_2019` to be less computationally expensive and scale better.

```{code-cell} ipython3
# Configure attention normalization
normalize_attention = RMSNorm(config.d_model, config.rms_norm_eps).to(device)

# Load pre-trained weights
load_state(normalize_attention, "normalize_attention", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Preserve residuals
residual = x

# Normalize attention inputs
x = normalize_attention(x)

x.shape
```


### Project Queries, Keys, Values

Next, we'll configure and then apply the linear projections {math}`\mathbf{W}_Q`, {math}`\mathbf{W}_K`, {math}`\mathbf{W}_V` that map input embeddings {math}`\mathbf{X}` to query, key, and value subspaces.

```{code-cell} ipython3
# Configure query, key, value projections
w_q = nn.Linear(
    in_features=config.d_model,
    out_features=config.n_heads * config.d_head,
    bias=False,
    device=device,
)
w_k = nn.Linear(
    in_features=config.d_model,
    out_features=config.n_kv_heads * config.d_head,
    bias=False,
    device=device,
)
w_v = nn.Linear(
    in_features=config.d_model,
    out_features=config.n_kv_heads * config.d_head,
    bias=False,
    device=device,
)

# Load pre-trained weights
load_state(w_q, "w_q", w_k, "w_k", w_v, "w_v", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Project embeddings to query, key, value spaces
q = w_q(x)
k = w_k(x)
v = w_v(x)

q.shape, k.shape, v.shape
```


### Split Attention Heads

Now that we've projected our embeddings to {math}`\mathbf{Q}`, {math}`\mathbf{K}`, and {math}`\mathbf{V}`, it's time to split up the attention heads.

```{code-cell} ipython3
def split_heads(x, n_heads):    
    return x.view(-1, n_heads, config.d_head).transpose(-3, -2)
```

```{code-cell} ipython3
# Split attention heads
q = split_heads(q, config.n_heads)
k = split_heads(k, config.n_kv_heads)
v = split_heads(v, config.n_kv_heads)

q.shape, k.shape, v.shape
```

```{code-cell} ipython3
# Expand key/value groups
reps = config.n_heads // config.n_kv_heads
k = k.repeat_interleave(reps, dim=0)
v = v.repeat_interleave(reps, dim=0)

q.shape, k.shape, v.shape
```

```{code-cell} ipython3
# Sanity check
assert q.shape == k.shape == v.shape
```

Let's take a quick moment to walk through what just happened. Starting with {math}`\mathbf{Q}`, we see the shape changed from {math}`22 \times 4096` to {math}`32 \times 22 \times 128`. The shape of {math}`\mathbf{K}` and {math}`\mathbf{V}` changed from {math}`22 \times 1024` to {math}`8 \times 22 \times 128` and then changed again to {math}`32 \times 22 \times 128`. But why? What is happening here?

The goal of GQA is to split each embedding's query, key, and value representation into chunks, expand the key / value representations to match the queries, distribute each chunk to an attention head, and then eventually recombine everything on the other side. Figures <a href="#figure7">7</a> to <a href="#figure9">9</a> illustrate the GQA process. In the first step, we have a query representation with a dimension of 16, a key representation with a dimension of 4, and a value representation also with dimension 4. 


```{figure} resources/gqa_step1.svg
:label: gqa-step1-fig

GQA - Step 1
```

Step 2 shows the representations after `split_heads`. Each color represents an attention head. We can see we have 8 query heads and 2 key/value heads.

```{figure} resources/gqa_step2.svg
:label: gqa-step2-fig

GQA - Step 2
```

Step 3 shows the representations after the key / value groups have been expanded. All of the attention heads have a full set of queries, keys, and values. If you look closely, you'll see the first group of attention heads all share the same keys and values {math}`\set{k_0, k_1, v_0, v_1}`. Similarly, the second group of attention heads all share the same keys and values {math}`\set{k_2, k_3, v_2, v_3}`.

```{figure} resources/gqa_step3.svg
:label: gqa-step3-fig

GQA - Step 3
```


### Encode Positions

Now that we've split queries, keys, and values into attention heads, the next step is to rotate the queries and keys using RoPE. Recall the compact, single embedding form of the RoPE transformation we looked at earlier:

```{math}
\mathbf{R} \mathbf{x} = 
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix}
\odot
\begin{bmatrix}
cos(m \theta_0) \\
cos(m \theta_0) \\
\vdots \\
cos(m \theta_{d/2-1}) \\
cos(m \theta_{d/2-1}) \\
\end{bmatrix}
+
\begin{bmatrix}
-x_1 \\
x_0 \\
\vdots \\
-x_{d-1} \\
x_{d-2} \\
\end{bmatrix}
\odot
\begin{bmatrix}
sin(m \theta_0) \\
sin(m \theta_0) \\
\vdots \\
sin(m \theta_{d/2-1}) \\
sin(m \theta_{d/2-1}) \\
\end{bmatrix}
```

We'll calculate the {math}`cos` and {math}`sin` vectors for every position {math}`m` and stack them in {math}`n \times d_{head}` matrices {math}`\mathbf{R}_{cos}` and {math}`\mathbf{R}_{sin}`.

Given {math}`\Theta`, {math}`d=d_{head}`, {math}`n=\text{sequence length}`:

```{math}
\begin{aligned}
\mathbf{R}_{cos} &=
\begin{bmatrix}
\cos(0 \theta_0) & \cos(0 \theta_0) & \dots & \cos(0 \theta_{d/2-1}) & \cos(0 \theta_{d/2-1}) \\
\vdots & \vdots & \dots & \vdots & \vdots \\
\cos((n-1) \theta_0) & \cos((n-1) \theta_0) & \dots & \cos((n-1) \theta_{d/2-1}) & \cos((n-1) \theta_{d/2-1}) \\
\end{bmatrix} \\
\quad \\
\mathbf{R}_{sin} &=
\begin{bmatrix}
\sin(0 \theta_0) & \sin(0 \theta_0) & \dots & \sin(0 \theta_{d/2-1}) & \sin(0 \theta_{d/2-1}) \\
\vdots & \vdots & \dots & \vdots & \vdots \\
\sin((n-1) \theta_0) & \sin((n-1) \theta_0) & \dots & \sin((n-1) \theta_{d/2-1}) & \sin((n-1) \theta_{d/2-1}) \\
\end{bmatrix}
\end{aligned}
```

where

```{math}
\theta_i = \frac{1}{\Theta^{2i/d}}, i \in [0, 1, \dots, d/2-1]
```

```{code-cell} ipython3
# Hyperparameters
base = config.rope_theta
d = config.d_head
```

```{code-cell} ipython3
# Calculate thetas
i = torch.arange(d // 2, device=device)
thetas = base ** (-2 * i / d)

thetas.shape
```

```{code-cell} ipython3
thetas
```

```{code-cell} ipython3
# Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
thetas = thetas.repeat_interleave(2)

thetas.shape
```

```{code-cell} ipython3
thetas
```

```{code-cell} ipython3
def rope_frequencies(n):
    """Compute RoPE cos and sin rotation matrices."""
    
    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = torch.stack([m*thetas for m in range(n)])
    
    # Apply cos, sin
    r_cos = torch.cos(theta_stack)
    r_sin = torch.sin(theta_stack)
    
    # Sanity check
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head

    return r_cos, r_sin

# Compute cos and sin rotation matrices
r_cos, r_sin = rope_frequencies(len(x))

r_cos.shape, r_sin.shape
```

```{code-cell} ipython3
r_cos
```

```{code-cell} ipython3
r_sin
```

Next, we'll define `rope_swap` to transform the embedding pairs to match the second column of {math}`\mathbf{R}`.

```{math}
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix}
\mapsto
\begin{bmatrix}
-x_1 \\
x_0 \\
\vdots \\
-x_{d-1} \\
x_{d-2} \\
\end{bmatrix}
```

```{code-cell} ipython3
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
```

We can finally combine all the pieces of the RoPE rotational transform!

```{code-cell} ipython3
def rope_rotate(x, r_cos, r_sin):
    """Rotate embeddings using RoPE transform."""
    
    return (x * r_cos) + (rope_swap(x) * r_sin)
```

```{code-cell} ipython3
# Encode positions by rotating queries and keys
q = rope_rotate(q, r_cos, r_sin)
k = rope_rotate(k, r_cos, r_sin)

q.shape, k.shape, v.shape
```


### Calculate Attention

We've finally reached the attention equation! First, we'll compute the attention mask and then put all the pieces together.

```{code-cell} ipython3
# Compute attention mask M
n = len(x)
mask = torch.ones(n, n, dtype=torch.bool, device=device).tril(diagonal=0)
m = torch.zeros(n, n, device=device).masked_fill_(mask.logical_not(), float("-inf"))

m.shape
```

```{code-cell} ipython3
m
```

```{code-cell} ipython3
# Compute attention for all heads in parallel
a = softmax(q @ k.transpose(-2, -1) / np.sqrt(config.d_head) + m, dim=-1) @ v

a.shape
```


### Recombine Attention Heads

Next, we'll reassemble the attention heads.

```{code-cell} ipython3
def combine_heads(x):
    return x.transpose(-3, -2).contiguous().view(-1, int(config.n_heads * config.d_head))
```

```{code-cell} ipython3
# Combine attention heads
a = combine_heads(a)

a.shape
```


### Project Outputs

Now that we've calculated the attention representation, we need to project the attention representation back to embedding space before we combine with {math}`\mathbf{X}`.

```{code-cell} ipython3
# Configure attention output projection
w_a = nn.Linear(
    in_features=config.d_model, 
    out_features=config.d_model,
    bias=False,
    device=device,
)

# Load pre-trained weights
load_state(w_a, "w_a", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Project attention embeddings back to model space
a = w_a(a)

a.shape
```


### Combine Outputs with Residuals

Here we simply add the attention representation {math}`\mathbf{A}` with the embeddings we started with right before normalization. This type of "residual learning" is critical in deep neural networks to ensure the gradients have a strong path from start to finish.

```{code-cell} ipython3
# Combine attention embeddings with residuals
x = residual + a

x.shape
```


## Feedforward Networks

Feedforward network (FFN) blocks capitalize on the contextual information added through attention, applying the non-linear transformation magic that neural networks are famous for.


### SwiGLU Activation

Llama's FFN blocks use an activation mechanism known as a Gated Linear Unit (GLU). Instead of passing the outputs through the activation function as a traditional MLP would, GLUs use the activation function as a gate that controls whether or not a linear projection of the inputs is allowed through. Similar to residual learning, the main goal here is to mitigate the vanishing gradient issues by providing a linear path for the gradients while retaining non-linear transformation capabilities. {cite:p}`dauphin_language_2016`

The specific approach Llama uses is called a SwiGLU ("swish" GLU) which uses a sigmoid activation as the gate. This basically gives you the strong gradient flow of a traditional ReLU with a fancier non-linear gate. {cite:t}`shazeer_glu_2020` demonstrated SwiGLU to perform better than a large range of alternatives.


### FFN Workflow

The following flowchart highlights the steps we'll walk through next. You can see the FFN block follows a similar, albeit simpler, process as the attention block.

```{figure} resources/ffn-flow.svg
:label: ffn-workflow-fig

Feedforward Network Workflow
```


### Normalize Inputs

Just like the attention block, we'll start by saving the residuals and normalizing the input embeddings using RMSNorm.

```{code-cell} ipython3
# Configure FFN normalization
normalize_ffn = RMSNorm(config.d_model, config.rms_norm_eps).to(device)

# Load pre-trained state
load_state(normalize_ffn, "normalize_ffn", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Preserve residuals
residual = x

# Normalize FFN inputs
x = normalize_ffn(x)

x.shape
```


### Transform

The following equation illustrates the SwiGLU FFN transformation from {math}`d_{model}`-dimensional embeddings {math}`\mathbf{X}` to {math}`d_{ffn}`-dimensional hidden states {math}`\mathbf{F}`.

```{math}
\mathbf{F} = \sigma(\mathbf{G}) \odot \mathbf{H}
```

where

* {math}`\mathbf{H}` is an {math}`n \times d_{ffn}` tensor of hidden states
* {math}`\mathbf{G}` is an {math}`n \times d_{ffn}` tensor of gates
* {math}`\sigma` is the sigmoid activation function (silu)
* {math}`\odot` is element-wise multiplication

This expands to

```{math}
\mathbf{F} = \sigma(\mathbf{W}_G \mathbf{X}) \odot \mathbf{W}_H \mathbf{X}
```

where

* {math}`\mathbf{W}_{H}`, {math}`\mathbf{W}_{G}` are the linear projections for hidden states and gates respectively.

```{code-cell} ipython3
# Configure SwiGLU FFN
w_h = nn.Linear(
    in_features=config.d_model,
    out_features=config.d_ffn,
    bias=False,
    device=device,
)
w_g = nn.Linear(
    in_features=config.d_model,
    out_features=config.d_ffn,
    bias=False,
    device=device,
)

# Load pre-trained weights
load_state(w_h, "w_h", w_g, "w_g", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Apply transform
f = silu(w_g(x)) * w_h(x)

f.shape
```


### Project Outputs

Now that we've calculated the FFN representation, we need to project it back to embedding space before we combine with {math}`\mathbf{X}`.

```{code-cell} ipython3
# Configure FFN output projection
w_f = nn.Linear(
    in_features=config.d_ffn,
    out_features=config.d_model,
    bias=False,
    device=device,
)

# Load pre-trained weights
load_state(w_f, "w_f", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Project FFN embeddings back to model space
f = w_f(f)

f.shape
```


### Combine Outputs with Residuals

Just like we did with attention, we combine FFN representation with the residual input embeddings.

```{code-cell} ipython3
# Combine FFN embeddings with residuals
x = residual + f

x.shape
```


## Stacking the Layers

Let's take a quick moment to recap. We've just finished walking through each step in a single decoder layer, tracing the embeddings through both Attention and FFN blocks. Next, we'll pull all these steps together and repeat them for each layer in the stack, gradually converting representations of individual words into representations of abstract semantic concepts. 

```{code-cell} ipython3
def context_layers(x):
    # Compute cos and sin rotation matrices
    r_cos, r_sin = rope_frequencies(len(x))

    # Apply layer logic in a loop
    for layer in range(config.n_layers):
    
        # Load pre-trained state for layer
        load_pretrained_state(layer)
    
        #
        # Attention
        #
    
        # Normalize attention inputs
        residual = x
        x = normalize_attention(x)
        
        # Project embeddings to query, key, value spaces
        q = w_q(x)
        k = w_k(x)
        v = w_v(x)
        
        # Split attention heads
        q = split_heads(q, config.n_heads)
        k = split_heads(k, config.n_kv_heads)
        v = split_heads(v, config.n_kv_heads)
        
        # Expand key/value groups
        reps = config.n_heads // config.n_kv_heads
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
        a = softmax(q @ k.transpose(-2, -1) / np.sqrt(config.d_head) + m, dim=-1) @ v
    
        # Combine attention heads
        a = combine_heads(a)
        
        # Project attention representations back to model space
        a = w_a(a)
        
        # Combine attention representations with residual embeddings
        x = residual + a
        
        #
        # FFN
        #
    
        # Normalize FFN inputs
        residual = x
        x = normalize_ffn(x)
    
        # Apply SwiGLU transform
        f = silu(w_g(x)) * w_h(x)
    
        # Project FFN representations back to model space
        f = w_f(f)
        
        # Combine FFN representations with residual embeddings
        x = residual + f

    return x
```

```{code-cell} ipython3
# Start over from initial tokens
x = torch.tensor(token_ids, device=device)

# Map tokens to embeddings
x = embeddings(x)

# Transform token embeddings to semantic embeddings
x = context_layers(x)

x.shape
```

```{code-cell} ipython3
x
```

# Head

After all that transforming, we're finally ready to predict the next token. The mechanism is simpler than you may think. Since the semantic embeddings have been cross-pollinated with semantic context, we can use the last embedding to represent the entire sequence. This gives us an ideal feature vector to feed into a traditional classifier with one label for each token in the vocabulary. We'll use softmax to generate a probability distribution over all the tokens, and then randomly sample from the most probable tokens.

```{figure} resources/head-flow.svg
:label: head-workflow-fig

Head Workflow
```


## Normalize Inputs

```{code-cell} ipython3
# Configure head normalization
normalize_head = RMSNorm(config.d_model, config.rms_norm_eps).to(device)

# Load pre-trained weights
load_state(normalize_head, "normalize_head", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Normalize head inputs
x = normalize_head(x)

x.shape
```


## Classify Sequence

Here we classify the entire sequence by projecting the last semantic embedding from model space back to token space, producing one logit for each token id in the vocabulary.

```{code-cell} ipython3
# Configure output projection
w_head = nn.Linear(
    in_features=config.d_model,
    out_features=config.vocab_size,
    bias=False,
    device=device,
)

# Load pre-trained weights
load_state(w_head, "w_head", checkpoint=checkpoint)
```

```{code-cell} ipython3
# Use last embedding to represent the entire sequence
x = x[-1]

# Project embedding from model space to token space
x = w_head(x)

x.shape
```

We can see `x` has been transformed from a matrix of embeddings into a single vector of logits that represent each of the 128k tokens in the vocabulary.


## Top Token

While LLMs have a reputation for being stochastic and unpredictable, it may surprise you to learn everything we've done up until has been deterministic. Our prompt always maps to the same token ids, the token ids to the same embeddings, and even the layers of attention and feedforward blocks are completely predictable. All of the randomness of an LLM comes from the sampling process we'll start next.

But first, let's test all of our hard work. If we've done everything right so far, then the highest ranked output token should be "Boston".

```{code-cell} ipython3
# Select top scoring token
token_id = x.argmax()

# Decode token
token = tokenizer.decode([token_id]).strip()

token
```

```{code-cell} ipython3
# Verify answer
assert token == "Boston"
```

Pretty cool huh?


## Sample Tokens

Even though the top token gave us the correct answer, LLMs don't work this way in practice. Instead of always predicting the top token, LLMs randomly sample from a group of top tokens.

:::{card}
*Sampling is where all of the LLM's creativity comes from.*
:::

Sampling avoids deterministic and repetitive outputs, making the responses feel more natural and human like. It also gives the model a mechanism to avoid getting stuck in a loop like "I am happy because I am happy because ..."

{ref}`token-sampling-fig` highlights the steps involved in token sampling. Starting with logits for every token in the vocabulary, we'll apply temperature, top k, and top p transformations to narrow the pool down to the "best" tokens before we randomly select from the remaining candidates.

It's worth noting that we're leaving Llama land and entering application land. Once the Llama model has produced the output logits, it's up to the application to decide what to do with them. We'll pull all of the application pieces together in a complete generator in the last section.

```{figure} resources/sample-flow.svg
:label: token-sampling-fig

Token Sampling
```


### Temperature 

Temperature controls the model's "creativity" by transforming the logits before we convert them to probabilities. A temperature setting of less than 1.0 reduces creativity by sharpening the probability distribution. A temperature setting greater than 1 increases creativity by flattening the probability distribution and increasing the chance of selecting different tokens.

```{code-cell} ipython3
# Default temperature comes from Hugging Face's meta-llama/Meta-Llama-3.1-8B-Instruct model
config.temperature
```

```{code-cell} ipython3
# Apply temperature
x = x / config.temperature
```


### Ranking

Next, we'll convert the logits to probabilities using softmax and then sort them in descending order to prepare for TopK and TopP filters.

```{code-cell} ipython3
# Convert logits to probabilities
probs = softmax(x)

# Sort probabilities in descending order
probs, indices = probs.sort(descending=True)
```


### Top K

The Top K filter simply keeps the `config.top_k` token probabilities.

```{code-cell} ipython3
# Default top_k comes from Hugging Face's meta-llama/Meta-Llama-3.1-8B-Instruct model
config.top_k
```

```{code-cell} ipython3
# Retain top k tokens
probs = probs[:config.top_k]
print(f"{len(probs)} of {len(x)} tokens remaining")
```


### Top P

The Top P filter calculates the cumulative sum of the token probabilities and then retains as many as needed to reach `config.top_p`.

```{code-cell} ipython3
# Hyperparameters
config.top_p
```

```{code-cell} ipython3
# Find cutoff where cumulative probability exceeds top_p
cumulative_mask = probs.cumsum(dim=-1) > config.top_p
threshold_index = torch.argmax(cumulative_mask).item()

# Only apply threshold if top_p was exceeded
if cumulative_mask.any():
    probs = probs[:threshold_index+1]

print(f"{len(probs)} of {len(x)} tokens remaining")
```


### Random Selection

Now that we've reduced the pool to the best tokens, we randomly select a token. We use `torch.multinomial` specifically because it respects the tokens' probabilities when selecting a value.

```{code-cell} ipython3
# Print remaining token pool
for i, prob in enumerate(probs):
    print(f"token id {indices[i]}, token '{tokenizer.decode([indices[i]])}', score {prob:0.3f}")
```

We can see here that the Top K and Top P filters have reduced the pool to a single option. Given our very specific prompt, even the randomly sampled answer should always be "Boston".

```{code-cell} ipython3
# Sample from remaining tokens weighted by probability
sampled_index = torch.multinomial(probs, 1)

# Convert sampled_index to original logits
token_id = indices[sampled_index]

# Decode token
token = tokenizer.decode([token_id]).strip()

token
```

```{code-cell} ipython3
# Verify answer
assert token == "Boston"
```


## Complete Head Stage

Next, we'll assemble all of the steps in the Head stage together before we take the generative loop for a spin in the next section.

```{code-cell} ipython3
def head(x):
    # Normalize head inputs
    x = normalize_head(x)
    
    # Use last embedding to represent the entire sequence
    x = x[-1]
    
    # Project outputs to token space
    x = w_head(x)

    #
    # Temperature
    #
    
    # Apply temperature
    x = x / config.temperature

    #
    # Ranking
    #
    
    # Convert logits to probabilities
    probs = softmax(x)
    
    # Sort probabilities in descending order
    probs, indices = probs.sort(descending=True)

    #
    # Top K
    #
    
    # Retain top k tokens
    probs = probs[:config.top_k]

    #
    # Top P
    #
    
    # Find cutoff where cumulative probability exceeds top_p
    cumulative_mask = probs.cumsum(dim=-1) > config.top_p
    threshold_index = torch.argmax(cumulative_mask).item()
    
    # Only apply threshold if top_p was exceeded
    if cumulative_mask.any():
        probs = probs[:threshold_index+1]

    #
    # Random Selection
    #
    
    # Sample from remaining tokens weighted by probability
    sampled_index = torch.multinomial(probs, 1)
    
    # Convert sampled_index to original logits
    token_id = indices[sampled_index]

    return token_id.item()
```


# Generator

We made it to the last section of the teardown. So far we've gone through every step from raw data to the first output token. In this section, we'll wrap the underlying Llama model in a generative loop that appends the output token to the end of the input sequence and repeats the whole process. The loop continues generating content one token at a time until the model predicts a "stop" token.

```{figure} resources/transformer-pipeline.svg
:label: text-generation-pipeline2-fig

Text Generation Pipeline
```

The entire end-to-end pipeline we started with is repeated above to remind us of the big picture. We'll run each of these stages including the autoregressive decoding loop at the bottom. Finally, we'll add a few extra utilities to make the prompts a lot easier to write and make the whole model easier to experiment with.

While the generative loop below is functional, it's not an efficient implementation. Instead of literally appending the output token to the `token_ids` list and repeating the entire process, it's common practice for production-grade generators to cache the keys and values from the first pass and then only pass the new token through the pipeline on the subsequent iterations, saving a huge amount of compute and memory. Maybe we can improve this version in a future post.

```{code-cell} ipython3
class Message(NamedTuple):
    role: str
    content: str
```

```{code-cell} ipython3
def prepare_messages(messages: Sequence[Message]):
    # Initialize prompt
    prompt = ""
    
    # Format each message
    for message in messages:
        prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
        prompt += message.content
        prompt += "<|eot_id|>"

    # Finish with the assistant role to prime the model's response
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt
```

```{code-cell} ipython3
def generate(messages: Sequence[Message]):
    # Format message prompt
    prompt = prepare_messages(messages)
    
    # Split raw text into tokens
    token_ids = tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")
    
    # Generate output until we get a stop token or we exceed max_output_tokens.
    for _ in range(config.max_completion_tokens):
        
        # Load token ids into a tensor
        x = torch.tensor(token_ids, device=device)
        
        # Map tokens to embeddings
        x = embeddings(x)
        
        # Transform token embeddings to semantic embeddings
        x = context_layers(x)
        
        # Head
        token_id = head(x)
        
        # Check stopping criteria
        if token_id in tokenizer.stop_tokens:
            break
    
        # Print token
        token = tokenizer.decode([token_id])
        stdout.write(token)
        
        # Append to end of sequence
        token_ids.append(token_id)
```

For the first experiment, let's stick with our original prompt but give the model more room to answer.

```{code-cell} ipython3
generate([
    Message(role="user", content="What is the capital of Massachusetts?")
])
```

Congrats! We just generated a full sentence. There are of course an unlimited number of experiments you could run. Let's try something a little more creative before we wrap up.

```{code-cell} ipython3
generate([
    Message(role="user", content="Write a haiku about fall in New England.")
])
```


# Recap

Congratulations! We dissected a state-of-the-art foundation model, cataloged the parts, and reassembled everything, tracing an inference from raw text to output token. While it wasn't a trivial undertaking, I hope you're walking away with a stronger understanding of the mechanisms driving the Generative AI revolution. I'm also incredibly grateful you stuck with me this far!

# References

```{bibliography}
:csl: site/themes/stickshift/assets/styles/chicago-author-date.csl
```
