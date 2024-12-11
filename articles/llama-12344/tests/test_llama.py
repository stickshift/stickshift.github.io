from llama.model import LlamaGenerator, load_checkpoint, load_config, load_tokenizer
from llama.tools import torch_device
import torch


def test_checkpoint():
    #
    # Givens
    #

    # I configured gpus
    device = torch_device()

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    #
    # Whens
    #

    # I load checkpoint
    checkpoint = load_checkpoint(config, map_location=device)

    #
    # Thens
    #

    # checkpoint should include Generator keys
    assert "model.embeddings.weight" in checkpoint

    for layer_id in range(config.n_layers):
        assert f"model.layers.{layer_id}.attention.normalize.weight" in checkpoint
        assert f"model.layers.{layer_id}.attention.w_queries.weight" in checkpoint
        assert f"model.layers.{layer_id}.attention.w_keys.weight" in checkpoint
        assert f"model.layers.{layer_id}.attention.w_values.weight" in checkpoint
        assert f"model.layers.{layer_id}.attention.w_output.weight" in checkpoint
        assert f"model.layers.{layer_id}.ffn.normalize.weight" in checkpoint
        assert f"model.layers.{layer_id}.ffn.w_input.weight" in checkpoint
        assert f"model.layers.{layer_id}.ffn.w_gate.weight" in checkpoint
        assert f"model.layers.{layer_id}.ffn.w_output.weight" in checkpoint

    assert "head.normalize.weight" in checkpoint
    assert "head.w_output.weight" in checkpoint

    # tensors should be loaded to device
    assert checkpoint["model.embeddings.weight"].device.type == device.type


def test_load_state_dict():
    #
    # Givens
    #

    # I configured gpus
    device = torch_device()

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I loaded checkpoint
    checkpoint = load_checkpoint(config, map_location=device)

    # I created a generator
    generator = LlamaGenerator(config, device=device)

    #
    # Whens
    #

    # I load state from checkpoint
    generator.load_state_dict(checkpoint)

    #
    # Thens
    #

    # state should match checkpoint
    assert torch.equal(generator.model.embeddings.get_parameter("weight"), checkpoint["model.embeddings.weight"])


def test_tokenizer():
    #
    # Givens
    #

    # I loaded config for Llama 3.2 3B checkpoint
    model_config = load_config("Llama3.2-3B")

    # I created a tokenizer
    tokenizer = load_tokenizer(model_config)

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I encode prompt
    token_ids = tokenizer.encode(prompt)

    #
    # Thens
    #

    # token_ids should be [128000, 7288, 13746, 22350]
    assert token_ids == [128000, 7288, 13746, 22350]

    #
    # Whens
    #

    # I decode last token id
    token = tokenizer.decode([token_ids[-1]])

    #
    # Thens
    #

    # token should be gamma
    assert token.strip() == "gamma"


def test_generate():
    #
    # Givens
    #

    # I configured gpus
    device = torch_device()

    # I loaded config for Llama 3.2 3B checkpoint
    config = load_config("Llama3.2-3B")

    # I created a tokenizer
    tokenizer = load_tokenizer(config)

    # I created a generator w/ token sampling disabled
    generator = LlamaGenerator(config, device, temperature=0)

    # I load state from checkpoint
    generator.load_state_dict(load_checkpoint(config, map_location=device))

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I split prompt into tokens
    token_ids = tokenizer.encode(prompt)

    # I generate next token
    token_id = next(generator(token_ids))

    #
    # Thens
    #

    # token should be "delta"
    assert tokenizer.decode([token_id]).strip() == "delta"
