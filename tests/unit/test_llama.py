import torch
from transformers import pipeline


def test_text_generation(device: torch.device):
    #
    # Givens
    #

    # I created llama 3 text generation pipeline
    transformer = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens=200,
        device=device,
    )

    # Prompt
    prompts = [
        {
            "role": "user",
            "content": "What is the capital of Massachusetts? Answer in one word."
        }
    ]

    #
    # Whens
    #

    # I generate answer
    response = transformer(prompts, return_full_text=False)
    answer = response[0]["generated_text"].strip()

    #
    # Thens
    #

    # The sentence should be classified as positive
    assert answer == "Boston."

