from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.net.residual import ResidualMLP
from spectralrl.utils.utils import at_least_ndim

from .network import RFFReward


# positional features
class PositionalFeature(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.dim = dim

    def forward(self, x):
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.to(freqs.dtype) @ freqs.unsqueeze(0)
        # x = x.squeeze()
        # x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierFeature(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("weight", torch.randn(1, dim // 2) * scale)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

# ================= noise schedules =================
def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)

def vp_beta_schedule(T: int = 1000):
    t = np.arange(1, T+1)
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

def generate_noise_schedule(
    noise_schedule: str="linear",
    num_noises: int=1000,
    beta_min: float=1e-4,
    beta_max: float=0.02,
    s: float=0.008
):
    if noise_schedule == "linear":
        betas = linear_beta_schedule(beta_min, beta_max, num_noises)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(s, num_noises)
    elif noise_schedule == "vp":
        betas = vp_beta_schedule(num_noises)
    else:
        raise NotImplementedError
    alphas = 1 - betas
    alphabars = np.cumprod(alphas, axis=0)
    return torch.as_tensor(betas, dtype=torch.float32), \
           torch.as_tensor(alphas, dtype=torch.float32), \
           torch.as_tensor(alphabars, dtype=torch.float32)


class DDPM(nn.Module):
    def __init__(self, cfg, state_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = cfg.feature_dim
        self.sample_steps = cfg.sample_steps
        self.x_min, self.x_max = cfg.x_min, cfg.x_max
        self.betas, self.alphas, self.alphabars = generate_noise_schedule(
            cfg.noise_schedule, cfg.sample_steps, cfg.beta_min, cfg.beta_max, cfg.s
        )
        self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)

        self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
            self.betas[..., None].to(self.device), \
            self.alphas[..., None].to(self.device), \
            self.alphabars[..., None].to(self.device), \
            self.alphabars_prev[..., None].to(self.device)

        self.mlp_s = nn.Sequential(
            nn.Linear(state_dim, cfg.embed_dim*2),
            nn.Mish(),
            nn.Linear(cfg.embed_dim*2, cfg.embed_dim),
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(action_dim, cfg.embed_dim*2),
            nn.Mish(),
            nn.Linear(cfg.embed_dim*2, cfg.embed_dim),
        )
        self.mlp_t = nn.Sequential(
            PositionalFeature(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim*2),
            nn.Mish(),
            nn.Linear(cfg.embed_dim*2, cfg.embed_dim),
        )
        self.mlp_psi = ResidualMLP(
            input_dim=cfg.embed_dim*2,
            output_dim=cfg.feature_dim,
            hidden_dims=cfg.psi_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )
        self.mlp_zeta = ResidualMLP(
            input_dim=state_dim+cfg.embed_dim,
            output_dim=state_dim*cfg.feature_dim,
            hidden_dims=cfg.zeta_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )
        self.reward = RFFReward(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.reward_hidden_dim,
        )

    def add_noise(self, x0):
        noise_idx = torch.randint(0, self.sample_steps, (x0.shape[0], )).to(self.device)
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def compute_loss(self, next_state, state, action, reward):
        x0 = next_state
        xt, t, eps = self.add_noise(x0)
        alphabars = self.alphabars[t]

        z_psi = self.forward_psi(s=state, a=action)
        z_zeta = self.forward_zeta(sp=xt.detach(), t=t.unsqueeze(-1))
        score = self.forward_score(z_psi=z_psi, z_zeta=z_zeta)
        diffusion_loss = (score - eps).pow(2).sum(-1).mean()
        if reward is not None:
            reward_pred = self.reward.forward(z_psi)
            reward_loss = (reward_pred - reward).pow(2).sum(-1).mean()
        else:
            reward_loss = torch.tensor(0.0)
        stats = ({
            "info/x0_l1_norm": x0.abs().mean(),
            "info/eps_l1_norm": eps.abs().mean(),
            "info/score_l1_norm": score.abs().mean(),
            "info/x0_mean": x0.mean(),
            "info/x0_std": x0.std(0).mean(),
        })
        return diffusion_loss, reward_loss, stats

    def sample(
        self,
        shapes,
        state,
        action,
        preserve_history=False
    ):
        info = {}

        def reverse(xt, t):
            z = torch.randn_like(xt)
            timestep = torch.full([xt.shape[0], ], t, dtype=torch.int64, device=self.device)
            score = self.forward(xt, timestep.unsqueeze(-1), state, action, psi=psi)
            sigma_t = 0
            if t > 0:
                sigma_t_square = self.betas[timestep]*(1-self.alphabars_prev[timestep]) / (1-self.alphabars[timestep])
                sigma_t_square = sigma_t_square.clip(1e-20)
                sigma_t = sigma_t_square.sqrt()
            # return 1. / self.alphas[timestep].sqrt() * (xt + self.betas[timestep] * score) + sigma_t * z
            return 1. / self.alphas[timestep].sqrt() * (xt - self.betas[timestep] / (1-self.alphabars[timestep]).sqrt() * score) + sigma_t * z

        xt = torch.randn(shapes, device=self.device)
        if self.factorized:
            psi = self.score.forward_psi(state, action)
        else:
            psi = None
        if preserve_history:
            info["sample_history"] = [xt, ]
        for t in reversed(range(0, self.sample_steps)):
            next_xt = reverse(xt, t)
            next_xt = torch.clip(next_xt, self.x_min, self.x_max)
            if preserve_history:
                info["sample_history"].append(next_xt)
            xt = next_xt

        return xt, info

    def forward_psi(self, s, a):
        s_ff = self.mlp_s(s)
        a_ff = self.mlp_a(a)
        x = torch.concat([s_ff, a_ff], dim=-1)
        x = self.mlp_psi(x)
        return x

    def forward_zeta(self, sp, t):
        t_ff = self.mlp_t(t)
        all = torch.concat([sp, t_ff], dim=-1)
        all = self.mlp_zeta(all)
        return all.reshape(-1, self.feature_dim, self.state_dim)

    def forward_score(self, s=None, a=None, sp=None, t=None, z_psi=None, z_zeta=None):
        if z_psi is None:
            z_psi = self.forward_psi(s=s, a=a)
        if z_zeta is None:
            z_zeta = self.forward_zeta(sp=sp, t=t)
        score = torch.bmm(z_psi.unsqueeze(1), z_zeta).squeeze(1)
        return score
