---
title: "Transformer Teardown: DistilBERT"
subtitle: "See the Code Behind Fundamental Transformer Concepts Like Embeddings, Residuals, and Multi-Head Self-Attention"
published: "2024-09-04"
banner: resources/banner.png
bibliography:
  - articles/distilbert1/references.bib
kernelspec:
  name: distilbert1
---

I loved taking things apart as a kid. Especially discarded electronics. I used to keep a pile of the circuit boards I scavenged in my closet. If I stacked them together the right way, I was convinced I could build my own C-3PO.

As an adult, I still like taking things apart. Methodically dissecting, cataloging, and rebuilding helps me wrap my brain around new technology. Especially the hardcore stuff like LLMs and the Transformers that power them.

While there are a million papers, blogs, and tutorials written on Transformers, I still find it challenging to map the abstract ideas from the research literature into concrete, actionable steps you can experiment with. My engineer's brain wants to "see the code" behind high level concepts like embeddings, residuals, and multi-head self-attention. Yes, it's easy to find open source Transformer implementations, but they're often overloaded with configuration settings to the point that the main ideas are completely obscured.

The goal of this post is to give you a stronger sense of the Transformer machinery powering the AI revolution. We'll dissect Hugging Face's default text-classification model, lay all the pieces on the table, and then trace a single inference through the stack from raw data to final prediction. We'll illustrate the main ideas from the Transformer literature with minimal, straightforward, working Python code. You may be surprised by how few steps are required!

All of the code for this post is available in [GitHub](https://github.com/stickshift/stickshift.github.io/blob/main/posts/2024-09-04-transformer-teardown/2024-09-04-transformer-teardown.ipynb) but it's a lot easier to [read with nbviewer](https://nbviewer.org/github/stickshift/stickshift.github.io/blob/main/posts/2024-09-04-transformer-teardown/2024-09-04-transformer-teardown.ipynb)!

```{code-cell} ipython3
---
tags: [remove-cell]
---
from itertools import islice
from typing import Any, Iterable, NamedTuple

from matplotlib import pyplot as plt
import numpy as np
from pytest import approx
import seaborn as sns
import torch
from torch import nn
from torch.nn.functional import relu, softmax
import transformers
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
class Config(NamedTuple):
    """Custom DistilBERT config."""

    vocab_size: int
    
    d_model: int
    
    n_heads: int
    
    d_head: int
    
    d_ffn: int
    
    n_layers: int
    
    n_labels: int
    
    max_sequence_length: int

    batch_size: int


def load_config(model) -> Config:
    """Load config from distilbert model."""
    config = model.distilbert.config

    return Config(**{
        "vocab_size": config.vocab_size,
        "d_model": config.dim,
        "n_heads": config.n_heads,
        "d_head": int(config.dim / config.n_heads),
        "d_ffn": config.hidden_dim,
        "n_layers": config.n_layers,
        "n_labels": config.num_labels,
        "max_sequence_length": config.max_position_embeddings,
        "batch_size": 1,
    })


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


def load_state(*args, layer=None):
    # Defaults
    layer = layer if layer is not None else 0

    for module, key in take(2, args):
        match key:
            case "value_embeddings":
                module.load_state_dict({
                    "weight": parameters["distilbert.embeddings.word_embeddings.weight"],
                })
            case "position_embeddings":
                module.load_state_dict({
                    "weight": parameters["distilbert.embeddings.position_embeddings.weight"],
                })
            case "normalize_embeddings":
                module.load_state_dict({
                    "weight": parameters["distilbert.embeddings.LayerNorm.weight"], 
                    "bias": parameters["distilbert.embeddings.LayerNorm.bias"],
                })
            case "queries":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.attention.q_lin.weight"],
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.attention.q_lin.bias"],
                })
            case "keys":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.attention.k_lin.weight"],
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.attention.k_lin.bias"],
                })
            case "values":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.attention.v_lin.weight"],
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.attention.v_lin.bias"],
                })
            case "outputs":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.attention.out_lin.weight"],
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.attention.out_lin.bias"],
                })
            case "normalize_attention":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.sa_layer_norm.weight"], 
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.sa_layer_norm.bias"],
                })
            case "ffn":
                module.load_state_dict({
                    "0.weight": parameters[f"distilbert.transformer.layer.{layer}.ffn.lin1.weight"], 
                    "0.bias": parameters[f"distilbert.transformer.layer.{layer}.ffn.lin1.bias"],
                    "2.weight": parameters[f"distilbert.transformer.layer.{layer}.ffn.lin2.weight"], 
                    "2.bias": parameters[f"distilbert.transformer.layer.{layer}.ffn.lin2.bias"],
                })
            case "normalize_ffn":
                module.load_state_dict({
                    "weight": parameters[f"distilbert.transformer.layer.{layer}.output_layer_norm.weight"], 
                    "bias": parameters[f"distilbert.transformer.layer.{layer}.output_layer_norm.bias"],
                })
            case "classifier":
                module.load_state_dict({
                    "0.weight": parameters["pre_classifier.weight"], 
                    "0.bias": parameters["pre_classifier.bias"],
                    "2.weight": parameters["classifier.weight"], 
                    "2.bias": parameters["classifier.bias"],
                })                


def load_pretrained_state(layer):    
    # Load pre-trained state
    load_state(
        queries, "queries", 
        keys, "keys", 
        values, "values", 
        outputs, "outputs", 
        normalize_attention, "normalize_attention",
        ffn, "ffn",
        normalize_ffn, "normalize_ffn",
        layer=layer,
    )
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
# Configure gpu
device = torch_device()
```

# Text Classification with DistilBERT

If you've worked with Transformers at all, I'm sure you're familiar with Hugging Face's collection of Python libraries as well as their endless repository of models and datasets. Throughout the post, we'll be working with Hugging Face's default text classification model DistilBERT. DistilBERT is a smaller, faster, lighter-weight version of the original BERT model that's easier to experiment with. We'll use the pre-trained model parameters from Hugging Face, but we'll implement the model's logic step-by-step using a slightly modified version of the actual DistilBERT PyTorch implementation from Hugging Face's `transformers` library.

Before we get into the implementation, let's start by running the entire process end-to-end using Hugging Face's high level `pipeline` API. The following cells create a complete text classification pipeline and then apply it to the sentence "I love ice cream". As you might expect, the model classifies the sentence as overwhelmingly positive. Over the rest of the post, we'll break this prediction down and recreate it one step at a time.

```{code-cell} ipython3
# Specify default DistilBERT model
model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Create off-the-shelf text classification transformer
generator = transformers.pipeline("text-classification", model=model, device=device)
```

```{code-cell} ipython3
generator("I love ice cream")
```

```{code-cell} ipython3
:tags: remove-cell

# Load model config and pre-trained parameters
config = load_config(generator.model)
parameters = generator.model.state_dict()
```

# Transformer Pipeline

The following diagram depicts a Transformer as a multi-stage pipeline. The Context stage at the center of the pipeline is where most of the magic happens. The stages before and after Context provide the extra machinery required to convert raw data into input embeddings and output embeddings into task-specific outputs. While we'll focus on text data, it's worth noting that the same stages can be applied to all data modalities including audio and images {cite:p}`xu_2023`.

```{figure} resources/transformer-pipeline.svg
:label: transformer-pipeline-fig

Transformer Pipeline
```

# Tokenize

The Tokenize stage is responsible for breaking raw data into a sequence of "tokens". While the word "token" is often associated with text processing, the Transformer literature extends this to other data modalities as well. Examples include patches of an image or segments of an audio recording. In fact, tokenization is seen as a core strength of the Transformer architecture because it allows Transformers to process different types of data using a single, universal approach {cite:p}`xu_2023`.

While tokenization is a general concept, the specific algorithms used are modality-specific. In this case, our transformer uses an algorithm known as "word-piece" {cite:p}`devlin_2019` to split raw text into a sequence of tokens. Next, special tokens are injected to mark the beginning and end of the sequence. Each token is then converted into an integer-encoded categorical value using a fixed token vocabulary, producing the final sequence of "input_ids" that are passed to the next stage.

Since our primary interest is in the Transformer layers that come later, we'll use Hugging Face's off-the-shelf tokenizer implementation here.

```{code-cell} ipython3
# Extract tokenizer from generator
tokenizer = generator.tokenizer

# Tokenize sentence
batch = tokenizer("I love ice cream", return_tensors="pt")

batch
```

Tokenizing "I love ice cream" generates the token sequence: `[101, 1045, 2293, 3256, 6949, 102]`. If we decode the integer-encoded values to see what each one represents, we can see the four words are represented by values `1045` to `6949`. The values `101` and `102` represent special tokens `[CLS]` and `[SEP]` that were added to mark the beginning and end of the sequence respectively.

```{code-cell} ipython3
[tokenizer.decode(input_id) for input_id in batch.input_ids[0]]
```

# Embeddings

The second stage in the Transformer pipeline converts each of the integer-encoded categorical values into an "embedding". Embeddings {cite:p}`bengio_2000` are the fundamental data structure of the Transformer architecture. The Transformer layers we'll look at in the next stage take embeddings as input, *transform* them, and produce embeddings as output. Embeddings predated Transformers by almost 2 decades and are a fascinating topic in their own right. But we'll save the embeddings deep dive for another post. For now, all we need to know is embeddings represent each token as a unique point in an n-dimensional vector space. The vector space coordinates are initialized randomly and then learned during training.

Similar to tokenization, the steps required to convert tokens into embeddings depend on the data modality. In BERT-based text transformers, the Embeddings stage is typically implemented using 2 lookup tables. The first lookup table maps the value of each token to a unique embedding vector. The second lookup table maps the position of each token to a unique embedding vector. The value and position embeddings are then added together to create the initial token embeddings.

```{figure} resources/embeddings.svg
:label: embeddings-fig
:width: 600px

Embeddings
```

Let's start with value embeddings. First, we initialize the value embeddings lookup table. Next, we read the values from the tokenizer output. Finally, we pass the token values to the lookup table to get unique embeddings for each value.

```{code-cell} ipython3
# Initialize value embeddings lookup table
value_embeddings = nn.Embedding(
    num_embeddings=config.vocab_size, 
    embedding_dim=config.d_model,
)

# Load pre-trained state
load_state(value_embeddings, "value_embeddings")

# Calculate token values
values = torch.squeeze(batch.input_ids)

[tokenizer.decode(input_id) for input_id in values]
```

```{code-cell} ipython3
# Map token values to embeddings
v = value_embeddings(values)

v.shape
```

```{code-cell} ipython3
# Show sample of value embeddings
v
```

Next, we'll follow a similar set of steps for the position embeddings. We'll start by initializing the position embeddings lookup table. Next, we'll calculate the positions from the tokenizer output. Finally, we pass the token positions to the lookup table to get unique embeddings for each position.

```{code-cell} ipython3
# Configure position embeddings lookup table
position_embeddings = nn.Embedding(
    num_embeddings=config.max_sequence_length, 
    embedding_dim=config.d_model,
)

# Load pre-trained state
load_state(position_embeddings, "position_embeddings")

# Calculate token positions
positions = torch.arange(values.size(0))

positions
```

```{code-cell} ipython3
# Map token positions to embeddings
p = position_embeddings(positions)

p.shape
```

```{code-cell} ipython3
# Show sample of position embeddings
p
```

Now that we have value and position embeddings, we add and normalize them to get the final "position-encoded token embeddings".

```{code-cell} ipython3
# Configure embeddings normalization
normalize_embeddings = nn.LayerNorm(
    normalized_shape=config.d_model, 
    eps=1e-12,
)

# Load pre-trained state
load_state(normalize_embeddings, "normalize_embeddings")

# Add and normalize value and position embeddings
x = normalize_embeddings(v + p)

x.shape
```

```{code-cell} ipython3
# Show sample of token embeddings
x
```

Congrats! You've converted the raw text "I love ice cream" into embeddings that encode both the token values and positions.

# Context

In the previous stage, we mapped the token values and positions to embeddings. But these embeddings represent the tokens in *isolation*. The Context stage is responsible for infusing each embedding with contextual signals drawn from the entire sequence. At a conceptual level, this should be intuitive. For example, the meaning of the word "ice" changes when you add "cream" after it.

```{figure} resources/contextualized-embeddings.svg
:label: contextualized-embeddings-fig
:width: 600px

Contextualized Embeddings
```

## Layers of Attention and FFNs

The Context stage works by passing the token embeddings through multiple layers of attention and feedforward blocks. The attention blocks focus on relationships between tokens, augmenting each embedding with information drawn from the surrounding embeddings. The feedforward blocks focus on individual tokens, transforming the contextual clues added by attention with the non-linear transformation magic neural networks are famous for.

The following diagram illustrates the stack of Transformer layers in the Context stage. The contents of each layer are identical. By arranging the layers in a stack, the model builds context in small increments similar to the hierarchical features in a CNN. The main differences between popular Transformer models such as BERT and GPT come down to how these layers are configured.

```{figure} resources/transformer-layers.svg
:label: transformer-layers-fig

Transformer Layers
```

As illustrated above, given input embeddings {math}`X`, we can define the output embeddings {math}`Z` for a single layer as:

```{math}
\begin{aligned}
Y &= Normalize(X + Attention(X)) \\
Z &= Normalize(Y + FFN(Y))
\end{aligned}
```

## Scaled Dot-Product Attention

The Attention block is the signature component of the Transformer architecture. It's also one of the most complicated and likely the least familiar when you're first learning about Transformers. We'll walk through the core attention algorithm described in the original "All You Need is Attention" paper by Vaswani et al. one step at a time. At the end of the Context section, we'll put all the pieces together.

{cite:t}`vaswani_2017` described their attention algorithm as Scaled Dot-Product Attention (SDPA) and defined the standard attention equation everyone cites:

```{math}
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_K}})V
```

## Queries, Keys, Values

The {math}`Q`, {math}`K`, and {math}`V` terms in the SDPA equation are "query", "key", and "value" matrices respectively. Each row in {math}`Q`, {math}`K`, and {math}`V` represents a token embedding that has been projected to distinct representation subspaces. Query embeddings represent selection criteria for the surrounding tokens that would add context to the current token definition. Key embeddings represent characteristics that satisfy the selection criteria. Value embeddings represent the contextual information one token transfers to another. Together, queries, keys, and values allow the attention mechanism to refine the representation of each token based on the surrounding tokens.

```{code-cell} ipython3
# Configure query, key, value projections
queries = nn.Linear(
    in_features=config.d_model, 
    out_features=config.d_model,
)
keys = nn.Linear(
    in_features=config.d_model,
    out_features=config.d_model,
)
values = nn.Linear(
    in_features=config.d_model, 
    out_features=config.d_model,
)

# Load pre-trained state
load_state(queries, "queries", keys, "keys", values, "values")

# Project token embeddings to query, key, and value spaces
q = queries(x)
k = keys(x)
v = values(x)

q.shape, k.shape, v.shape
```

We can see the projections generated unique query, key, and value embeddings for each of the 6 tokens `['[CLS]', 'i', 'love', 'ice', 'cream', '[SEP]']`.

## Attention Weights

Now that we have {math}`Q`, {math}`K`, and {math}`V`, we can delve into the SDPA equation itself. For each input embedding, SDPA calculates a weighted sum of the value projections for all the tokens in the sequence. We already saw the value projections are represented by {math}`V`. The weights are represented by the softmax term:

```{math}
\begin{aligned}
softmax(\frac{QK^T}{\sqrt{d_K}})
\end{aligned}
```

I wouldn't hold it against you if it's not immediately obvious what we get here. To see what's happening, let's break this down even further.

First, the {math}`QK^T` term calculates a {math}`d_Q \times d_K` matrix of the dot products of each query embedding with every key embedding. To see why, imagine we have 2 token embeddings of length 3. Using matrix multiplication, we end up with a {math}`2 \times 2` matrix where each element {math}`w_{ij}` represents the dot product of query {math}`i` with key {math}`j`.

```{math}
\begin{aligned}
QK^T
&=
\begin{bmatrix}
q_{00} & q_{01} & q_{02} \\
q_{10} & q_{11} & q_{12}
\end{bmatrix}
\begin{bmatrix}
k_{00} & k_{10} \\
k_{01} & k_{11} \\
k_{02} & k_{12}
\end{bmatrix}
=
\begin{bmatrix}
w_{00} & w_{01} \\
w_{10} & w_{11}
\end{bmatrix} \\
\text{where } w_{ij} &= row(Q, i) \cdot row(K, j)
\end{aligned}
```

```{code-cell} ipython3
# Calculate similarity between Q and K
w = q @ k.transpose(-2, -1)

w.shape
```

Second, the {math}`1/\sqrt{d_K}` term scales the dot products down to avoid pushing the softmax function into regions with very small gradients.

```{code-cell} ipython3
w /= np.sqrt(config.d_head)

w.shape
```

Finally, the softmax function normalizes the weights across the keys.

```{code-cell} ipython3
# Normalize weights across keys
w = softmax(w, dim=-1)

w
```

```{code-cell} ipython3
:tags: hide-input

# Plot weights for each query
_, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,5), gridspec_kw={"wspace": 0.5, "hspace": 0.5})
for i in range(6):
    ax = axs[i//3][i%3]
    sns.scatterplot(x=np.arange(len(w[i])), y=w[i].detach(), ax=ax)
    ax.set_title(f"Query {i}")
    ax.set_xlabel(f"Keys")
    ax.set_ylabel(f"Weight")
```

## Attention Output

Now that we have the attention weights, we can apply them to the values. This will give us a weighted sum of contextual information. However, the answer is still in "value space". Before we combine them with the token embeddings, we'll project them back to "model space".

```{code-cell} ipython3
# Compute weighted combination of values
a = w @ v

a.shape
```

```{code-cell} ipython3
# Configure output projection
outputs = nn.Linear(
    in_features=config.d_model, 
    out_features=config.d_model,
)

# Load pre-trained state
load_state(outputs, "outputs")

# Project attention embeddings back to model space
a = outputs(a)

a.shape
```

## Multi-Head Attention

At this point, we've walked through the core SDPA algorithm step-by-step. However, we're not quite done. Vaswani et al. realized there are more than one set of relationships involved in transferring context across tokens. A single application of SDPA would effectively water these down by averaging them together. The solution is to apply SDPA multiple times on separate query, key, and value embeddings. Each of these is referred to as an "attention head". Each head is isolated, leaving it free to learn distinct relational structures.

```{code-cell} ipython3
def split_heads(x):
    return x.view(-1, config.n_heads, config.d_head).transpose(-3, -2)

def combine_heads(x):
    return x.transpose(-3, -2).contiguous().view(-1, int(config.n_heads * config.d_head))
```

```{code-cell} ipython3
# Render query, key, value dimensions before we split
q.shape, k.shape, v.shape
```

```{code-cell} ipython3
# Split queries, keys, values into separate heads
q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

q.shape, k.shape, v.shape
```

We can see that the queries, keys, and values have been split into 12 heads. Each of the original 768-element query, key, and value embeddings is now 64 elements long.

Next, let's recompute the attention embeddings.

```{code-cell} ipython3
# Compute attention for all heads in parallel
a = softmax(q @ k.transpose(-2, -1) / np.sqrt(config.d_head), dim=-1) @ v

a.shape
```

While the attention code is the same, you can see the attention values are still split into heads. Next, we'll recombine them before applying the final output projection.

```{code-cell} ipython3
# Recombine heads
a = combine_heads(a)

a.shape
```

```{code-cell} ipython3
# Project attention embeddings back to model space
a = outputs(a)

a.shape
```

## Add and Normalize

Before we get to the FFN, we combine the attention embeddings with input embeddings the same way we combined the value and position embeddings.

```{code-cell} ipython3
# Configure attention normalization
normalize_attention = nn.LayerNorm(
    normalized_shape=config.d_model, 
    eps=1e-12,
)

# Load pre-trained state
load_state(normalize_attention, "normalize_attention")

# Combine attention with input embeddings
y = normalize_attention(x + a)

y.shape
```

## FFN

The FFN block is a straightforward fully connected multi-layer perceptron.

```{code-cell} ipython3
# Configure FFN
ffn = nn.Sequential(
    nn.Linear(in_features=config.d_model, out_features=config.d_ffn),
    nn.GELU(),
    nn.Linear(in_features=config.d_ffn, out_features=config.d_model),
)

# Load pre-trained state
load_state(ffn, "ffn")

# Transform attention outputs
f = ffn(y)

f.shape
```

## Add and Normalize

Next, we combine the transformed embeddings with the attention embeddings.

```{code-cell} ipython3
# Configure attention normalization
normalize_ffn = nn.LayerNorm(
    normalized_shape=config.d_model,
    eps=1e-12,
)

# Load pre-trained state
load_state(normalize_ffn, "normalize_ffn")

z = normalize_ffn(y + f)

z.shape
```

Quick recap... We just finished going through a single Transformer layer. Given input embeddings {math}`X`, we calculated and added attention embeddings to get {math}`Y`, and then calculated and added transformed embeddings to get {math}`Z`.

```{math}
\begin{aligned}
Y &= Normalize(X + Attention(X)) \\
Z &= Normalize(Y + FFN(Y))
\end{aligned}
```

## Stacking the Layers

Next, we combine all of the steps and repeat for each layer in the stack. While you would normally create a stack of torch modules, instead we run the layers in a loop to make it easier to see what's happening.

```{code-cell} ipython3
# Initialize loop w/ initial input embeddings
z_i = x

# Apply layer logic in a loop
for layer in range(config.n_layers):
    
    # Use previous layer's outputs as inputs
    x_i = z_i

    # Load pre-trained state for layer
    load_pretrained_state(layer)

    #
    # Attention
    #
    
    # Project x_i to query, key, and value spaces
    q_i = queries(x_i)
    k_i = keys(x_i)
    v_i = values(x_i)
    
    # Split q, k, v into separate attention heads
    q_i = split_heads(q_i)
    k_i = split_heads(k_i)
    v_i = split_heads(v_i)

    # Compute attention for all heads in parallel
    w_i = softmax(
        q_i @ k_i.transpose(-2, -1) / np.sqrt(config.d_head), 
        dim=-1,
    )
    a_i = w_i @ v_i
    
    # Recombine attention heads
    a_i = combine_heads(a_i)
    
    # Project attention embeddings back to model space
    a_i = outputs(a_i)

    # Add and normalize
    y_i = normalize_attention(x_i + a_i)

    #
    # FFN
    #

    # Transform attention
    f_i = ffn(y_i)

    # Add and normalize
    z_i = normalize_ffn(y_i + f_i)

# Save outputs from last layer
z = z_i
```

```{code-cell} ipython3
z
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
# Sanity check
hiddens = generator.model.distilbert(input_ids=batch.input_ids.to(device)).last_hidden_state.squeeze().to("cpu")
assert torch.allclose(z, hiddens, atol=1e-5)
```

# Head

As the final stage in the Transformer pipeline, Head maps the contextualized embeddings to task-specific predictions. In our case, the Head stage is responsible for turning the contextualized embeddings into a binary classifier that predicts whether the original text contains positive or negative sentiments. This sounds like a straightforward neural network output layer until you realize that instead of one set of features, we have a sequence of features. And the length of the sequence is arbitrary. How do you connect an arbitrary length sequence of feature vectors to an output layer?

The trick is hiding in our contextualized embeddings. Each input embedding represents a single token in isolation. But the output embeddings have been infused with information from all of the tokens. This is why it's common practice to simply take the first output embedding and drop the rest. The first embedding represents the start of sequence marker `[CLS]`. Since the `[CLS]` marker token is added to every sequence, the first *input* embedding is always the same. In contrast, the first *output* embedding uniquely represents this specific sequence.

If we take the first output embedding to represent the entire sequence, then we have a single feature vector that's easy to connect to any task-specific output layer we need.

```{code-cell} ipython3
# Use [CLS] embedding to represent the entire sequence
features = z[0]

features.shape
```

```{code-cell} ipython3
# Configure classifier head
classifier = nn.Sequential(
    nn.Linear(in_features=config.d_model, out_features=config.d_model),
    nn.ReLU(),
    nn.Linear(in_features=config.d_model, out_features=config.n_labels),
)

# Load pre-trained state
load_state(classifier, "classifier")

# Classify features
prediction1 = torch.softmax(classifier(features), dim=-1)[1].item()
prediction1
```

```{code-cell} ipython3
# Verify custom results match off-the-shelf ones
prediction2 = generator("I love ice cream")[0]["score"]
prediction2
```

```{code-cell} ipython3
assert prediction1 == approx(prediction2)
```

# Recap

We did it! We dissected an off-the-shelf, production-grade Transformer model, cataloged all the parts, and reassembled everything, tracing a single inference from the raw text "I love ice cream" to a positive sentiment prediction of `0.99`. Not only did we get the same positive prediction, we got the exact same answer as Hugging Face's production PyTorch code! That should give you confidence at least that we didn't leave any steps out.

I hope you learned something about Transformers. In the very least, I hope they're a little less intimidating. Go take something else apart!

# References

```{bibliography}
:csl: site/themes/stickshift/assets/styles/chicago-author-date.csl
```
