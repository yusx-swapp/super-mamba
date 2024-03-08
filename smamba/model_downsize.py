from functools import partial
import numpy as np
import copy
from torch import nn
import torch
import time

from typing import List
from mamba_ssm.modules.mamba_simple import Mamba, Block, RMSNorm
from mamba_ssm.models.mixer_seq_simple import create_block
from .elasticity import create_block

__all__ = [
    "bert_module_handler",
    "vit_module_handler",
    "vit_peft_module_handler",
    "arc_config_sampler",
    "sam_module_handler",
    "T5_module_handler",
    "distilbert_module_handler",
]


def copy_weights_to_subnet(subnet, org_model):
    """
    Copies the weights from original foundation model to scaled subnet where the parameter names match.
    Only the overlapping parts of the weights are copied when the dimensions in the subnet
    are less than or equal to those in the larger model.

    Parameters:
    subnet (torch.nn.Module): The smaller model to which the weights will be copied.
    org_model (torch.nn.Module): The foundation model from which the weights will be sourced.

    Usage:
    This function is useful in extract subnet from pre-trained foundation model scenarios where a smaller model is initialized
    with weights from certain layers of a larger, pre-trained model.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]
            if all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                # Create a slice object for each dimension to copy the corresponding weights
                slices = tuple(
                    slice(0, min(sm_dim, lg_dim))
                    for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
                )
                sm_param.data.copy_(lg_param.data[slices])


def check_weight_copy_correctness(subnet, org_model):
    """
    Checks if the weights have been correctly copied from the larger model to the smaller model.

    Parameters:
    smaller_model (torch.nn.Module): The smaller model with copied weights.
    larger_model (torch.nn.Module): The larger model from which the weights were sourced.

    Returns:
    bool: True if the weights are correctly copied, False otherwise.

    Usage:
    Useful for verifying the correctness of a weight copying process in model adaptation or transfer learning.
    """

    for sm_param_name, sm_param in subnet.named_parameters():
        if sm_param_name in dict(org_model.named_parameters()):
            lg_param = dict(org_model.named_parameters())[sm_param_name]

            # Compare shapes
            if not all(
                sm_dim <= lg_dim
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            ):
                return False

            # Compare values
            slices = tuple(
                slice(0, min(sm_dim, lg_dim))
                for sm_dim, lg_dim in zip(sm_param.shape, lg_param.shape)
            )
            if not torch.all(sm_param == lg_param[slices]):
                return False

    return True


def mamba_base_module_handler(
    block,
    arc,
    d_model,
    ssm_cfg,
    rms_norm,
    residual_in_fp32,
    fused_add_norm,
    layer_idx,
    factory_kwargs,
):
    new_block = create_block(
        d_model=d_model,
        d_inner=arc["d_inner"],
        ssm_cfg=ssm_cfg,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        layer_idx=layer_idx,
        **factory_kwargs,
    )
    copy_weights_to_subnet(new_block, block)
    return new_block


def mamba_model_handler(
    model,
    arc,
    device="cuda",
    dtype=torch.float32,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    new_model = copy.deepcopy(model)
    for idx, (layer, layer_arc) in enumerate(zip(model.backbone.layers, arc)):
        new_block = mamba_base_module_handler(
            layer,
            arc[layer_arc],
            model.config.d_model,
            model.config.ssm_cfg,
            model.config.rms_norm,
            model.config.residual_in_fp32,
            model.config.fused_add_norm,
            idx,
            factory_kwargs,
        )
        new_model.backbone.layers[idx] = new_block
    return new_model


if __name__ == "__main__":
    block = create_block(
        d_model=768,
        ssm_cfg={},
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ).to("cuda")

    x = torch.randn(2, 64, 768).to("cuda")

    h, residule = block(x)
    print(h.shape, residule.shape)

    for name, param in block.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print("Done")
    pass
