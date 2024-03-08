import time
import math
import torch
import copy
import torch.nn as nn
from functools import partial
from einops import repeat
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba, Block, RMSNorm


class Elasticity:
    """Elasticity configuration class for Mamba"""

    def __init__(self, **kwargs):
        self.depth_elasticity = kwargs.get("depth_elasticity", False)
        self.width_elasticity = kwargs.get("width_elasticity", True)
        self.n_layer = kwargs.get("n_layer", 48)

        self.d_inner_space = kwargs.get("d_inner_space", [1024, 1536, 2048, 2560, 3072])
        self.d_inner_space.sort()

        self.depth_space = kwargs.get(
            "depth_space", [i for i in range(0, self.n_layer)]
        )
        self.depth_space.sort()

        np.random.seed(int(time.time()))

    def get_elasticity(self) -> dict:
        return copy.deepcopy(self.__dict__)

    def __repr__(self):
        return f"Depth Elasticity: {self.depth_elasticity}, Width Elasticity: {self.width_elasticity}, Number of Layers: {self.n_layer}"

    def arc_sampling(self, smallest=False, largest=False):

        assert (
            smallest or largest or (not smallest and not largest)
        ), "Only one of smallest or largest can be True"

        arc = {}

        if self.depth_elasticity:
            raise NotImplementedError

        self.activate_layers = self.depth_space

        if self.width_elasticity:

            for i in self.activate_layers:

                arc[f"layer_{i}"] = {"d_inner": np.random.choice(self.d_inner_space)}

                if smallest:
                    arc[f"layer_{i}"] = {"d_inner": self.d_inner_space[0]}
                elif largest:
                    arc[f"layer_{i}"] = {"d_inner": self.d_inner_space[-1]}

        return arc

    def smallest_arc(self):
        return self.arc_sampling(smallest=True)

    def largest_arc(self):
        return self.arc_sampling(largest=True)


class ElasticMamba(Mamba):
    """The ElasticMamba class is a subclass of Mamba that allows for elastic width and depth configurations."""

    def __init__(
        self,
        d_model,
        d_inner=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(d_model, **factory_kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = d_inner
        if not d_inner:
            self.d_inner = int(self.expand * self.d_model)

        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )


def create_block(
    d_model,
    d_inner=None,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        ElasticMamba, d_inner=d_inner, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
