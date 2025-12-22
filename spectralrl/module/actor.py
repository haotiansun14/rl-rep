from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from spectralrl.module.net.mlp import MLP, EnsembleMLP
from spectralrl.utils.distributions import SquashedNormal

ModuleType = Type[nn.Module]

class BaseActor(nn.Module):
    """
    BaseActor interface.
    """
    def __init__(self) -> Any:
        super().__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass of the actor, only handles the inference of internal model.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, obs: torch.Tensor, *args, **kwargs) -> Any:
        """Sampling procedure.

        Parameters
        ----------
        obs :  The observation, shoule be torch.Tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, obs, action, *args, **kwargs) -> Any:
        """Evaluate the log_prob of the action.

        obs :  The observation, shoule be torch.Tensor.
        action :  The action for evaluation, shoule be torch.Tensor with the sample size as `obs`.
        """
        raise NotImplementedError


class GaussianActor(BaseActor):
    """
    Gaussian Actor, which maps the given obs to a parameterized Gaussian Distribution over the action space.

    Parameters
    ----------
    backend :  The preprocessing backend of the actor, which is used to extract vectorized features from the raw input.
    input_dim :  The dimensions of input (the output of backend module).
    output_dim :  The dimension of actor's output.
    reparameterize : Whether to use the reparameterization trick when sampling.
    conditioned_logstd :  Whether the logstd is conditioned on the observation.
    fix_logstd :  If not None, the logstd will be set to this value and fixed (un-learnable).
    logstd_min: The minimum value of logstd. Default is -20.
    logstd_max: The maximum value of logstd. Default is 2.
    logstd_hard_clip: Whether or not to hard-clip the logstd. If True, then logstd = clip(logstd_out, logstd_min, logstd_max); otherwise logstd = frac{tanh(logstd_out)+1}{2}*(logstd_max-logstd_min) + logstd_min.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reparameterize: bool=True,
        conditioned_logstd: bool=True,
        fix_logstd: Optional[float]=None,
        logstd_min: float = -20.0,
        logstd_max: float = 2.0,
        logstd_hard_clip: bool=True,
        device: Union[str, int, torch.device]="cpu",
        *,
        ensemble_size: int = 1,
        hidden_dims: Union[int, Sequence[int]]=[],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__()

        self.actor_type = "GaussianActor"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparameterize = reparameterize
        self.device = device
        self.logstd_hard_clip = logstd_hard_clip

        if fix_logstd is not None:
            self._logstd_is_layer = False
            self.register_buffer("logstd", torch.tensor(fix_logstd))
        elif not conditioned_logstd:
            self._logstd_is_layer = False
            self.logstd = nn.Parameter(torch.zeros([self.output_dim]), requires_grad=True)
        else:
            self._logstd_is_layer = True
            self.output_dim = output_dim = 2*output_dim

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

        self.register_buffer("logstd_min", torch.tensor(logstd_min, dtype=torch.float32))
        self.register_buffer("logstd_max", torch.tensor(logstd_max, dtype=torch.float32))

    def forward(self, input: torch.Tensor):
        out = self.output_layer(input)
        if self._logstd_is_layer:
            mean, logstd = torch.split(out, self.output_dim // 2, dim=-1)
        else:
            mean = out
            logstd = self.logstd.broadcast_to(mean.shape)
        if self.logstd_hard_clip:
            logstd = torch.clip(logstd, min=self.logstd_min, max=self.logstd_max)
        else:
            logstd = self.logstd_min + (torch.tanh(logstd)+1)/2*(self.logstd_max - self.logstd_min)
        return mean, logstd

    def sample(self, obs: torch.Tensor, deterministic: bool=False, return_mean_logstd: bool=False, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Sampling procedure. The action is sampled from a Gaussian distribution.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        deterministic :  Whether to sample or return the mean action.
        return_mean_logstd :  Whether to return the mean and logstd of the Normal distribution.

        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict.
        """
        mean, logstd = self(obs)
        dist = Normal(mean, logstd.exp())
        if deterministic:
            action, logprob = dist.mean, None
        elif self.reparameterize:
            action = dist.rsample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)

        info = {"mean": mean, "logstd": logstd} if return_mean_logstd else {}
        return action, logprob, info

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, return_dist: bool=False, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Evaluate the action at given obs.

        Parameters
        ----------
        obs : The observation, should be torch.Tensor.
        action :  The action, shoild torch.Tensor.
        return_dist :  Whether to return the action distrbution at obs in info dict.

        Returns
        -------
        (torch.Tensor, Dict) :  The log-probability of action at obs and the info dict.

        :param state: state of the environment.
        :param action: action to be evaluated.
        """
        mean, logstd = self(obs)
        dist = Normal(mean, logstd.exp())
        info = {"dist": dist} if return_dist else {}
        return dist.log_prob(action).sum(-1, keepdim=True), info


class SquashedGaussianActor(GaussianActor):
    """
    Squashed Gaussian Actor, which maps the given obs to a parameterized Gaussian Distribution, followed by a Tanh transformation to squash the action sample to [-1, 1].

    Parameters
    ----------
    backend :  The preprocessing backend of the actor, which is used to extract vectorized features from the raw input.
    input_dim :  The dimensions of input (the output of backend module).
    output_dim :  The dimension of actor's output.
    reparameterize : Whether to use the reparameterization trick when sampling.
    conditioned_logstd :  Whether the logstd is conditioned on the observation.
    fix_logstd :  If not None, the logstd will be set to this value and fixed (un-learnable).
    logstd_min: The minimum value of logstd. Default is -20.
    logstd_max: The maximum value of logstd. Default is 2.
    logstd_hard_clip: Whether or not to hard-clip the logstd. If True, then logstd = clip(logstd_out, logstd_min, logstd_max); otherwise logstd = frac{tanh(logstd_out)+1}{2}*(logstd_max-logstd_min) + logstd_min.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        reparameterize: bool = True,
        conditioned_logstd: bool = True,
        fix_logstd: Optional[float] = None,
        logstd_min: float = -20.0,
        logstd_max: float = 2.0,
        logstd_hard_clip: bool=True,
        device: Union[str, int, torch.device]="cpu",
        *,
        ensemble_size: int = 1,
        hidden_dims: Union[int, Sequence[int]]=[],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__(
            input_dim, output_dim, reparameterize, conditioned_logstd, fix_logstd, logstd_min, logstd_max, logstd_hard_clip, device,
            ensemble_size=ensemble_size,
            hidden_dims=hidden_dims,
            norm_layer=norm_layer,
            activation=activation,
            dropout=dropout,
            share_hidden_layer=share_hidden_layer
        )
        self.actor_type = "SquashedGaussianActor"

    def sample(self, obs: torch.Tensor, deterministic: bool=False, return_mean_logstd=False, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Sampling procedure. The action is sampled from a Tanh-transformed Gaussian distribution.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.
        deterministic :  Whether to sample or return the mean action.
        return_mean_logstd :  Whether to return the mean and logstd of the TanhNormal distribution.

        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict.
        """
        mean, logstd = self.forward(obs)
        dist = SquashedNormal(mean, logstd.exp())
        if deterministic:
            action, logprob = dist.tanh_mean, None
        elif self.reparameterize:
            action, raw_action = dist.rsample(return_raw=True)
            logprob = dist.log_prob(raw_action, pre_tanh_value=True).sum(-1, keepdim=True)
        else:
            action, raw_action = dist.sample(return_raw=True)
            logprob = dist.log_prob(raw_action, pre_tanh_value=True).sum(-1, keepdim=True)

        info = {"mean": mean, "logstd": logstd} if return_mean_logstd else {}
        return action, logprob, info

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, return_dist: bool=False, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Evaluate the action at given obs.

        Parameters
        ----------
        obs : The observation, should be torch.Tensor.
        action :  The action, shoild torch.Tensor.
        return_dist :  Whether to return the action distrbution at obs in info dict.

        Returns
        -------
        (torch.Tensor, Dict) :  The log-probability of action at obs and the info dict.

        :param state: state of the environment.
        :param action: action to be evaluated.
        """
        mean, logstd = self.forward(obs)
        dist = SquashedNormal(mean, logstd.exp())
        info = {"dist": dist} if return_dist else False
        return dist.log_prob(action).sum(-1, keepdim=True), info


class DeterministicActor(BaseActor):
    """
    Deterministic Actor, which maps the given obs to a deterministic action.

    Parameters
    ----------
    input_dim :  The dimensions of input (the output of backend module).
    output_dim :  The dimension of actor's output.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: Union[str, int, torch.device]="cpu",
        *,
        ensemble_size: int = 1,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__()
        self.actor_type = "DeterministicActor"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims.copy()
        self.device = device

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

    def forward(self, input: torch.Tensor):
        return self.output_layer(input)

    def sample(self, obs: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Sampling procedure, note that in DeterministicActor we don't do any operation on the sample.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.

        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict.
        """
        return self(obs), None, {}

    def evaluate(self, *args, **kwargs) -> Any:
        """
        Evaluate the log_prob of the action. Note that this actor does not support evaluation.
        """
        raise NotImplementedError("Evaluation shouldn't be called for DeterministicActor.")


class SquashedDeterministicActor(DeterministicActor):
    """
    Squashed Deterministic Actor, which maps the given obs to a deterministic action squashed into [-1, 1] by tanh.

    Parameters
    ----------
    input_dim :  The dimensions of input (the output of backend module).
    output_dim :  The dimension of actor's output.
    device :  The device which the model runs on. Default is cpu.
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: Union[str, int, torch.device]="cpu",
        *,
        ensemble_size: int = 1,
        hidden_dims: Union[int, Sequence[int]]=[],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        share_hidden_layer: Union[Sequence[bool], bool] = False,
    ) -> None:
        super().__init__(input_dim, output_dim, device, ensemble_size=ensemble_size, hidden_dims=hidden_dims, norm_layer=norm_layer, activation=activation, dropout=dropout, share_hidden_layer=share_hidden_layer)
        self.actor_type = "SqushedDeterministicActor"

    def sample(self, obs: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Sampling procedure. The action is squashed into [-1, 1] by tanh.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.

        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict.
        """
        action_prev_tanh = super().forward(obs)
        return torch.tanh(action_prev_tanh), None, {}
