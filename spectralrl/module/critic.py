from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from spectralrl.module.net.basic import weight_init
from spectralrl.module.net.mlp import MLP, EnsembleMLP

ModuleType = Type[nn.Module]

class EnsembleQ(nn.Module):
    """
    A vanilla critic module, which can be used as Q(s, a) or V(s).

    Parameters
    ----------
    input_dim :  The dimensions of input.
    output_dim :  The dimension of critic's output.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int=1,
        device: Union[str, int, torch.device] = "cpu",
        *,
        ensemble_size: int=1,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__()
        self.critic_type = "Critic"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.ensemble_size = ensemble_size

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim,
                output_dim = output_dim,
                hidden_dims = hidden_dims,
                norm_layer = norm_layer,
                activation = activation,
                dropout = dropout,
                device = device,
                ensemble_size = ensemble_size,
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, *args, **kwargs) -> torch.Tensor:
        """Compute the Q-value (when action is given) or V-value (when action is None).

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        action :  The action, should be torch.Tensor.

        Returns
        -------
        torch.Tensor :  Q(s, a) or V(s).
        """
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        return self.output_layer(obs)
