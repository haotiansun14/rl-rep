import math
from typing import Any, List, Optional, Type

import torch
import torch.nn as nn

ModuleType = Type[nn.Module]

def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinear):
        for i in range(m.ensemble_size):
            nn.init.orthogonal_(m.weight.data[..., i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

def miniblock(
    input_dim: int,
    output_dim: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    dropout: Optional[ModuleType] = None,
    linear_layer: ModuleType = nn.Linear,
    *args,
    **kwargs
) -> List[nn.Module]:
    """
    Construct a miniblock with given input and output. It is possible to specify norm layer, activation, and dropout for the constructed miniblock.

    Parameters
    ----------
    input_dim :  Number of input features..
    output_dim :  Number of output features. Default is 0.
    norm_layer :  Module to use for normalization. When not specified or set to True, nn.LayerNorm is used.
    activation :  Module to use for activation. When not specified or set to True, nn.ReLU is used.
    dropout :  Dropout rate. Default is None.
    linear_layer :  Module to use for linear layer. Default is nn.Linear.

    Returns
    -------
    List of modules for miniblock.
    """

    layers: List[nn.Module] = [linear_layer(input_dim, output_dim, *args, **kwargs)]
    if norm_layer is not None:
        if isinstance(norm_layer, nn.Module):
            layers += [norm_layer(output_dim)]
        else:
            layers += [nn.LayerNorm(output_dim)]
    if activation is not None:
        layers += [activation()]
    if dropout is not None and dropout > 0:
        layers += [nn.Dropout(dropout)]
    return layers


class EnsembleLinear(nn.Module):
    """
    An linear module for concurrent forwarding, which can be used for ensemble purpose.

    Parameters
    ----------
    in_features :  Number of input features.
    out_features :  Number of output features.
    ensemble_size :  Ensemble size. Default is 1.
    bias :  Whether to add bias or not. Default is True.
    device :  Device to use for parameters.
    dtype :  Data type to use for parameter.
    """
    def __init__(
        self,
        in_features,
        out_features,
        ensemble_size: int = 1,
        share_input: bool=True,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.share_input = share_input
        self.add_bias = bias
        self.register_parameter("weight", torch.nn.Parameter(torch.empty([in_features, out_features, ensemble_size], **factory_kwargs)))
        if bias:
            self.register_parameter("bias", torch.nn.Parameter(torch.empty([out_features, ensemble_size], **factory_kwargs)))
        else:
            self.register_buffer("bias", torch.zeros([out_features, ensemble_size], **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            torch.nn.init.kaiming_uniform_(self.weight[..., i], a=math.sqrt(5))
        if self.add_bias:
            for i in range(self.ensemble_size):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[..., i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[..., i], -bound, bound)

    def forward(self, input: torch.Tensor):
        if self.share_input:
            res = torch.einsum('...j,jkb->...kb', input, self.weight) + self.bias
        else:
            res = torch.einsum('b...j,jkb->...kb', input, self.weight) + self.bias
        return torch.einsum('...b->b...', res)

    def __repr__(self):
        return f"EnsembleLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.add_bias})"
