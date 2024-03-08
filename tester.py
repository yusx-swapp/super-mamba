from smamba.elasticity import Elasticity
from smamba.model_downsize import mamba_model_handler
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Block
import torch
from mamba_ssm.models.mixer_seq_simple import create_block


import torch

import transformers
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from smamba import create_block
from smamba.modeling_smamba import SuperMamba


def elasticity_config_test():
    """
    Test Elasticity Config
    Status: Passed
    """
    elasticity = Elasticity()
    arc = elasticity.arc_sampling(smallest=True)
    print(arc)

    arc = elasticity.arc_sampling(largest=True)
    print(arc)

    arc = elasticity.arc_sampling()
    print(arc)


def generate_test():
    """
    Test Generate
    Status: Not Passed
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4B", device="cuda")
    input = "I love eating ice cream, because"

    tokenized_input = tokenizer(input, return_tensors="pt")
    tokenized_input.to("cuda")

    out = model.generate(
        tokenized_input["input_ids"],
        max_length=50,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        # return_dict_in_generate=True,
        output_scores=True,
    )
    print("==" * 50, "original model", "==" * 50)
    print(tokenizer.decode(out[0]))

    elasticity = Elasticity(d_inner_space=[1024, 1536, 2048, 2560, 3072])
    arc = elasticity.arc_sampling()
    subnet = mamba_model_handler(model, arc)
    subnet.to("cuda")

    out = subnet.generate(
        tokenized_input["input_ids"],
        max_length=50,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        # return_dict_in_generate=True,
        output_scores=True,
    )


def debug_generate():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4B", device="cuda")
    input = "I love eating ice cream, because"

    tokenized_input = tokenizer(input, return_tensors="pt")
    tokenized_input.to("cuda")
    elasticity = Elasticity(d_inner_space=[1024, 1536, 2048, 2560, 3072])
    arc = elasticity.arc_sampling()
    subnet = mamba_model_handler(model, arc)
    subnet.to("cuda")

    print(subnet)
    print("Start inference")
    out = subnet(tokenized_input["input_ids"]).logits.argmax(-1)
    print(tokenizer.decode(out[0]))

    print("Start generate")
    out = subnet.generate(
        tokenized_input["input_ids"],
        max_length=50,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        # return_dict_in_generate=True,
        output_scores=True,
    )


def inference_test():
    """
    Test Inference
    Status: passed
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4B", device="cuda")
    input = "I love eating ice cream, because"

    tokenized_input = tokenizer(input, return_tensors="pt")
    tokenized_input.to("cuda")

    print("=" * 50, "inference original model", "=" * 50)
    out = model(tokenized_input["input_ids"]).logits.argmax(-1)
    print(tokenizer.decode(out[0]))

    print("=" * 50, "inference subnet model", "=" * 50)
    supernet = SuperMamba(model, None)
    # elasticity = Elasticity(d_inner_space=[1024, 1536, 2048, 2560, 3072])
    # arc = elasticity.arc_sampling()
    # subnet = mamba_model_handler(model, arc)
    supernet.random_resource_aware_model()
    supernet.to("cuda")
    out = supernet(tokenized_input["input_ids"]).logits.argmax(-1)
    print(tokenizer.decode(out[0]))

    print("=" * 50, "subnet vs original model", "=" * 50)
    print(supernet)
    # print(model)


if __name__ == "__main__":
    inference_test()
