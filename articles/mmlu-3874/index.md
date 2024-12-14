---
title: "Transformer Teardown: Exploring Order Dependency Bias"
subtitle: "Lorem ipsum dolor sit amet."
published: "2024-12-14"
banner: resources/banner.png
bibliography:
  - articles/mmlu-3874/references.bib
kernelspec:
  name: mmlu-3874
---

```{code-cell} python
:tags: remove-cell

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import os
from pathlib import Path
from random import sample
import sys


from IPython.display import display, HTML
from matplotlib import pyplot as plt
from pandas import DataFrame
import torch
from torch import nn
from torch.nn import functional as F

from article.llama import load_config, load_parameters, load_tokenizer, LlamaModel, LlamaHead
from article.tools import torch_device


# ------------------------------------------------------------------------------
# Local Functions
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Configure
# ------------------------------------------------------------------------------

project_path = Path(os.environ["PROJECT_ROOT"])
datasets_path = project_path / ".build" / "datasets"
```

Do you feel like your LLM gives you inconsistent answers? Ever been frustrated trying to recreate a previous chat only to have your LLM take you down a completely different path? *Well you may not be wrong...* 

Multiple researchers have demonstrated LLMs of all shapes and sizes are sensitive to the order information is presented in the prompt. Even small changes to the order of tokens in a sequence can make a large difference in the model's response.

Like hallucinations, order dependency is a significant obstacle to user acceptance and broader adoption of AI solutions. This is because order dependency leads to inconsistent model outputs, making models seem unreliable and eroding users' trust. For AI solutions to deliver on their promises, they need to be more robust. We can't accept healthcare models that change a diagnosis based on the order a patient's labs are entered. Or financial models that approve or deny loan applications based on who is first or last in line.


In [our last Transformer Teardown post](/articles/llama-12344/), we built a lightweight Llama research and development kit. In this post, we'll use our toolkit to explore the language model order dependency problem. We'll start by evaluating models using multiple choice questions (MCQs) from the MMLU benchmark. Next, we'll see if we can reproduce findings from {cite:t}`pezeshkpour_large_2023` and {cite:t}`zheng_large_2024` where they showed that reshuffling the order of the options dramatically changed the models' scores. 

Source code for the accompanying `article` package is available in GitHub: [Article Package](https://github.com/stickshift/stickshift.github.io/tree/main/articles/mmlu-3874/src/article)

# MMLU Benchmark

Massive Multitask Language Understanding (MMLU) {cite:p}`hendrycks_measuring_2020` is a popular benchmark for evaluating language models' world knowledge and problem solving abilities. The MMLU dataset contains 14,042 multiple choice questions (MCQs) from 57 categories including mathematics, history, biology, and business. Each question has 4 options (A, B, C, D) and one correct answer. In addition, each category includes 5 example questions designed for few shot experiments. 

When MMLU was first published in 2020, only the largest GPT models could do better than random guessing. By 2024, multiple models from OpenAI, Anthropic, Meta, and Tencent have all published MMLU accuracies over 88%. Initial reports of Open AI's o1 are greater than 92%. While these results are certainly impressive, we're all making an implicit assumption that they're robust. It would be quite disconcerting if an insignificant change to the benchmark caused a model's score to shift by 10% to 20%!

:::{card}
*We're all making an implicit assumption that a model's benchmark scores are robust.*
:::

## Load Dataset

To get started, let's download [the MMLU dataset](https://github.com/hendrycks/test) and inspect a sample of the questions.

```{code-cell} python
from article.mmlu import download_dataset, load_dataset, display_questions
```

```{code-cell} python
# Configure path
dataset_path = datasets_path / "mmlu"

# Download dataset
download_dataset(dataset_path)
```

```{code-cell} python
# Load dataset
examples, questions = load_dataset(dataset_path)
categories = {q.category for q in questions}
print(f"Loaded {len(examples)} examples, {len(questions)} questions, {len(categories)} categories")
```

```{code-cell} python
display_questions(sample(questions, k=3))
```

```{code-cell} python
pass
```

```{code-cell} python
pass
```

