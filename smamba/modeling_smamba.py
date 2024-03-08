""" Official Implementation
One Foundation Model Fits All: Single-stage Foundation Model Training with Zero-shot Deployment
"""

import copy
import os
from typing import Any
import torch
from torch.nn import Parameter

from torch import nn

from .model_downsize import mamba_model_handler
from .utils import calculate_params, load_dict_from_file
from .elasticity import Elasticity


class SuperMamba(nn.Module):
    def __init__(self, model, elastic_config=None) -> None:
        super(SuperMamba, self).__init__()
        self.model = model
        self.activate_model = model
        self.total_params = calculate_params(model=model)

        if hasattr(self.model.config, "elastic_config"):
            elastic_config = self.model.config.elastic_config

        if not elastic_config:
            # set defalt search space configuration (this is defalt setting for bert)
            print(
                f"[Warning]: No elastic configuration provides. Set to the defalt elastic space {elastic_config}."
            )
        elif isinstance(elastic_config, str):
            elastic_config = load_dict_from_file(elastic_config)

        # assert isinstance(
        #     elastic_config, dict
        # ), "Invalid elastic_config, expect input a dictionary or file path"
        if not elastic_config:
            self.elasticity = Elasticity(d_inner_space=[1024, 1536, 2048, 2560, 3072])
        else:
            self.elasticity = Elasticity(**elastic_config)

        self.model.config.elastic_config = self.elasticity.get_elasticity()

        self.local_grads = []
        self.alphas = []
        self._pre_global_grad = None

    def forward(self, *args, **kwargs):
        return self.activate_model(*args, **kwargs)

    def random_resource_aware_model(self):
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        arc = self.elasticity.arc_sampling()
        subnetwork, total_params = self.resource_aware_model(arc)

        return subnetwork, total_params, arc

    def smallest_model(self):
        """Return the smallest model in the elastic space

        Returns:
            - subnetwork (nn.Module): The smallest model in the elastic space
            - params (int): The number of parameters in million of the smallest model
            - arc_config (dict): The configuration of the smallest model
        """
        arc = self.elasticity.smallest_arc()
        subnetwork, params = self.resource_aware_model(arc)
        return subnetwork, params, arc

    def largest_model(self):
        arc = self.elasticity.smallest_arc()
        return copy.deepcopy(self.model), self.total_params, arc

    def resource_aware_model(self, arc_config):
        subnet = mamba_model_handler(self.model, arc_config)
        self.set_activate_model(subnet)
        return subnet, calculate_params(subnet)

    def set_activate_model(self, model):
        """Set the activate model based on the arc_config

        Args:
            arc_config (dict): The configuration of the model
        """
        self.activate_model = model

    def salient_parameter_prioritization(self, metric="l1_norm"):

        # self.model = salient_parameter_prioritization(self.model, metric)
        pass

    def grad_accumulate(self, local_grad, alpha=None):
        self.local_grads.append(local_grad)
        self.alphas.append(alpha)

    def apply_grad(self, grad):
        """Apply the gradients to the full-size model

        Args:
            grad (dict): Trained downsized model gradients
        """
        self.model.to("cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                local_grad = grad[name].cpu()
                slices = tuple(
                    slice(0, min(sm_dim, lg_dim))
                    for sm_dim, lg_dim in zip(local_grad.shape, param.shape)
                )
                if self._pre_global_grad:
                    param[slice] -= (
                        0.9 * local_grad + 0.1 * self._pre_global_grad[name][slice]
                    )
                else:
                    param[slices] -= local_grad

    def apply_accumulate_grad(self, beta=0.5):
        self.grad_normalization()

        self.model.to("cpu")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                for local_grad, alpha in zip(self.local_grads, self.alphas):
                    local_param_grad = local_grad[name].cpu()
                    slices = tuple(
                        slice(0, min(sm_dim, lg_dim))
                        for sm_dim, lg_dim in zip(local_param_grad.shape, param.shape)
                    )
                    param[slices] -= (
                        local_param_grad * alpha / sum(self.alphas)
                    ) * beta

        self.local_grads.clear()
        self.alphas.clear()

    def grad_normalization(self):
        """Normalize the gradients via previous epoch's gradients"""
        pass

    def save_ckpt(self, dir):
        self.model.save_pretrained(os.path.join(dir))

    def load_ckpt(self, dir):
        self.model = self.model.from_pretrained(dir)
        # check the the existance of self.model.config.elastic_config
        assert hasattr(
            self.model.config, "elastic_config"
        ), "No elastic configuration found in the model config file. Please check the config file."
