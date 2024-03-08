from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Block
import torch
from mamba_ssm.models.mixer_seq_simple import create_block


import torch

import transformers
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from smamba import create_block, copy_weights_to_subnet

# batch, length, dim = 2, 64, 768

# block = create_block(
#     d_model=dim,
#     ssm_cfg={"expand": 2},
#     norm_epsilon=1e-5,
#     rms_norm=False,
#     residual_in_fp32=False,
#     fused_add_norm=False,
#     layer_idx=None,
#     device=None,
#     dtype=None,
# ).to("cuda")

# print(block)
# x = torch.randn(batch, length, dim).to("cuda")

# h, residule = block(x)
# print(h.shape, residule.shape)

# for name, param in block.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.shape)
import torch.nn.functional as F

x = torch.randn(1, 4096, 7)
print(F.pad(x, (-3, 0)).shape)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4B", device="cuda")
input = "I love eating ice cream, because"
tokenized_input = tokenizer(input, return_tensors="pt")
tokenized_input.to("cuda")

# out = model.generate(
#     tokenized_input["input_ids"],
#     max_length=50,
#     top_k=50,
#     top_p=0.95,
#     temperature=1.0,
#     # return_dict_in_generate=True,
#     output_scores=True,
# )
# # print(out.keys())
# print(tokenizer.decode(out[0]))

# print(model.backbone.layers[1])

# print(model.config)
factory_kwargs = {"device": "cuda"}
block = create_block(
    model.config.d_model,
    d_inner=4095,
    ssm_cfg=model.config.ssm_cfg,
    rms_norm=model.config.rms_norm,
    residual_in_fp32=model.config.residual_in_fp32,
    fused_add_norm=model.config.fused_add_norm,
    layer_idx=1,
    **factory_kwargs,
)
# print(block)
copy_weights_to_subnet(block, model.backbone.layers[20])

model.backbone.layers[0] = block
# print(model)
# out = model.generate(
#     tokenized_input["input_ids"],
#     max_length=50,
#     top_k=50,
#     top_p=0.95,
#     temperature=1.0,
#     # return_dict_in_generate=True,
#     output_scores=True,
# )
out = model(tokenized_input["input_ids"]).logits.argmax(-1)
print(out)
print(tokenizer.decode(out[0]))
