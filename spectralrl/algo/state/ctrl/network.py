import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.module.net.mlp import MLP
from spectralrl.module.net.residual import ResidualMLP
from spectralrl.utils.utils import make_target, sync_target


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
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


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


class FactorizedInfoNCE(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        feature_dim,
        phi_hidden_dims,
        mu_hidden_dims,
        reward_hidden_dim,
        num_noises=0,
        device="cpu"
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device
        if num_noises > 0:
            self.use_noise_perturbation = True
            self.num_noises = num_noises
            self.betas, self.alphas, self.alphabars = generate_noise_schedule(
                "vp", num_noises,
            )
            self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)
            self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
                self.betas[..., None].to(self.device), \
                self.alphas[..., None].to(self.device), \
                self.alphabars[..., None].to(self.device), \
                self.alphabars_prev[..., None].to(self.device)
        else:
            self.use_noise_perturbation = False

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128)
        )
        self.mlp_phi = ResidualMLP(
            input_dim=obs_dim+action_dim,
            output_dim=feature_dim,
            hidden_dims=phi_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device
        )
        self.mlp_mu = ResidualMLP(
            input_dim=obs_dim+128,
            output_dim=feature_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device
        )
        self.rff_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, reward_hidden_dim, learnable=True),
        )
        self.reward_net = nn.Sequential(
            nn.Linear(2*reward_hidden_dim, reward_hidden_dim),
            nn.LayerNorm(reward_hidden_dim),
            nn.ELU(),
            nn.Linear(reward_hidden_dim, 1)
        )

    def forward_phi(self, s, a):
        x = torch.concat([s, a], dim=-1)
        x = self.mlp_phi(x)
        return x

    def compute_logits(self, s, a, sp, z_phi=None):
        LB = s.shape[0]
        RB = sp.shape[0]
        if z_phi is None:
            z_phi = self.forward_phi(s, a) # (LB, z_dim)
        if self.use_noise_perturbation:
            sp = sp.unsqueeze(0).repeat(self.num_noises, 1, 1)
            t = torch.arange(0, self.num_noises).to(self.device)
            t = t.repeat_interleave(RB).reshape(self.num_noises, RB)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp)
            xt = alphabars.sqrt() * sp + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)
        else:
            xt = sp.unsqueeze(0)
            t = None
        z_mu = self.forward_mu(xt, t) # (N, RB, z_dim)
        z_phi = z_phi.unsqueeze(0).repeat(z_mu.shape[0], 1, 1) # (N, LB, z_dim)
        logits = torch.bmm(z_phi, z_mu.transpose(-1, -2))      # (N, LB, RB)
        return logits

    def forward_mu(self, sp, t=None):
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = torch.concat([sp, t_ff], dim=-1)
        sp = self.mlp_mu(sp)
        return torch.nn.functional.tanh(sp)

    def compute_reward(self, z_phi):
        return self.reward_net(self.rff_layer(z_phi))

    def compute_feature(self, s, a):
        return self.forward_phi(s, a)


class RFFLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        learnable: bool = True
    ):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.layer = nn.Linear(feature_dim, hidden_dim)
        else:
            self.register_buffer("noise", torch.randn([feature_dim, hidden_dim], requires_grad=False))

    def forward(self, x):
        if self.learnable:
            x = self.layer(x)
            return torch.concat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            return torch.sin(x @ self.noise)


class RFFCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int
    ):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.net2 = nn.Sequential(
            nn.LayerNorm(feature_dim),
            RFFLayer(feature_dim, hidden_dim, learnable=True),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        q1 = self.net1(x)
        q2 = self.net2(x)
        return torch.stack([q1, q2], dim=0)
