from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from spectralrl.module.critic import EnsembleQ
from spectralrl.module.net.mlp import MLP


class MLPEncDec(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = [],
        deterministic: bool = True,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        logstd_min: float = -20.0,
        logstd_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim if self.deterministic else 2*output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            norm_layer=norm_layer
        )
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

    def forward(self, x):
        x = self.mlp(x)
        if self.deterministic:
            return x
        else:
            mean, logstd = torch.chunk(x, 2, dim=-1)
            logstd = torch.clip(logstd, min=self.logstd_min, max=self.logstd_max)
            return mean, logstd

    def sample(self, x, deterministic=False, return_mean_logstd=False):
        if self.deterministic:
            return self.forward(x)
        else:
            mean, logstd = self.forward(x)
            dist = Normal(mean, logstd.exp())
            if deterministic:
                s, logprob = dist.mean(), None
            else:
                s = dist.rsample()
                logprob = dist.log_prob(s).sum(-1, keepdim=True)
            if return_mean_logstd:
                return s, logprob, mean, logstd
            else:
                return s, logprob


class Encoder(MLPEncDec):
    def __init__(
        self,
        feature_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.LayerNorm,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
    ):
        super().__init__(
            input_dim=obs_dim+action_dim+obs_dim,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            deterministic=False,
            norm_layer=norm_layer,
            activation=activation,
        )

    def forward(self, obs, action, next_obs):
        x = torch.concat([obs, action, next_obs], axis=-1)
        return super().forward(x)

    def sample(self, obs, action, next_obs, deterministic=False, return_mean_logstd=False):
        if self.deterministic:
            return self.forward(obs, action, next_obs)
        else:
            mean, logstd = self.forward(obs, action, next_obs)
            dist = Normal(mean, logstd.exp())
            if deterministic:
                s, logprob = dist.mean(), None
            else:
                s = dist.rsample()
                logprob = dist.log_prob(s).sum(-1, keepdim=True)
            if return_mean_logstd:
                return s, logprob, mean, logstd
            else:
                return s, logprob


class Decoder(MLPEncDec):
    def __init__(
        self,
        feature_dim: int,
        obs_dim: int,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.LayerNorm,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
    ) -> None:
        super().__init__(
            input_dim=feature_dim,
            output_dim=obs_dim+1,
            hidden_dims=hidden_dims,
            deterministic=True,
            norm_layer=norm_layer,
            activation=activation
        )

    def forward(self, x):
        x = super().forward(x)
        return x[..., :-1], x[..., -1:]

    def sample(self, x):
        return self.forward(x)


class GaussianFeature(MLPEncDec):
    def __init__(
        self,
        feature_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = [],
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.LayerNorm,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
    ):
        super().__init__(
            input_dim=obs_dim+action_dim,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            deterministic=False,
            norm_layer=norm_layer,
            activation=activation,
        )

    def forward(self, obs, action):
        x = torch.concat([obs, action], axis=-1)
        return super().forward(x)

    def sample(self, obs, action, deterministic=False, return_mean_logstd=False):
        if self.deterministic:
            return self.forward(obs, action)
        else:
            mean, logstd = self.forward(obs, action)
            dist = Normal(mean, logstd.exp())
            if deterministic:
                s, logprob = dist.mean(), None
            else:
                s = dist.rsample()
                logprob = dist.log_prob(s).sum(-1, keepdim=True)
            if return_mean_logstd:
                return s, logprob, mean, logstd
            else:
                return s, logprob


class Mean(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class LVRepCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_noise: int=20,
        hidden_dim: int=256,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        # simulate sampling from Gaussian
        self.num_noise = num_noise
        self.register_buffer("noise", torch.randn([self.num_noise, feature_dim], requires_grad=False, device=device))

        self.net1 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            Mean(dim=1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            Mean(dim=1),
        )

    def forward(self, mean, logstd):
        std = logstd.exp()
        # B, D = mean.shape
        x = mean[:, None, :] + std[:, None, :] * self.noise[None, :, :]

        q1 = self.net1(x)
        q2 = self.net2(x)
        return torch.stack([q1, q2], dim=0)
