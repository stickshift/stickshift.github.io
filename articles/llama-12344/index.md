---
title: "Transformer Teardown: Minimal Llama Toolkit"
subtitle: ""
published: "2024-09-21"
banner: resources/banner.png
bibliography:
  - articles/llama1/references.bib
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: llama-12344
  language: python
  name: llama-12344
---

* Why build your own toolkit?
* Text Generation Pipeline
* Generator Design Pattern
* 

# Transformer Teardown: Minimal Llama Toolkit

In [the last Transformer Teardown post](https://stickshift.github.io/articles/llama1/), we tore down and rebuilt Meta's Llama LLM, gaining a hands-on, close-up view of the machinery inside a state-of-the-art generative transformer. The goal of this post is to leverage what we learned to create a minimal toolkit for experimenting with Llama models. While there are plenty of open source Llama implementations out there, they're often complicated and overloaded with configuration settings to the point that the main ideas are completely obscured. Building your own minimal Llama stack will do wonders for your knowledge of Transformer fundamentals. Not only that, but you'll walk away with a valuable toolkit for running your own research experiments. We'll be sure to revisit this idea in a future post.

# Text Generation Pipeline

{ref}`transformer-pipeline-fig` illustrates the main stages of a generative Transformer we reviewed in the last post. Raw text is split into tokens; tokens are mapped to embeddings; token embeddings are transformed into semantic embeddings through multiple layers of attention and feedforward networks; semantic embeddings are used to predict the next token in the sequence; the predicted token is fed back into the pipeline and the process repeats.

```{figure} resources/transformer-pipeline.svg
:label: transformer-pipeline-fig

Transformer Pipeline
```




