import torch
from transformers import pipeline


def test_text_generation(device: torch.device):
    #
    # Givens
    #

    # I created llama 3 text generation pipeline
    transformer = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)

    #
    # Whens
    #

    # I process 'What is the capital of Massachusetts?'
    transformer("What is the capital of Massachusetts?")

    #
    # Thens
    #

    # The sentence should be classified as positive
    pass
