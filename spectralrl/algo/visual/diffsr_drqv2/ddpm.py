from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.utils.utils import at_least_ndim

from .nn_idql import IDQLFactorizedScoreNet, IDQLScoreNet


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
    def __init__(self, args, state_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sample_steps = args.sample_steps
        self.x_min, self.x_max = args.x_min, args.x_max
        self.betas, self.alphas, self.alphabars = generate_noise_schedule(
            args.noise_schedule, args.sample_steps, args.beta_min, args.beta_max, args.s
        )
        self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.0)

        self.betas, self.alphas, self.alphabars, self.alphabars_prev = \
            self.betas[..., None].to(self.device), \
            self.alphas[..., None].to(self.device), \
            self.alphabars[..., None].to(self.device), \
            self.alphabars_prev[..., None].to(self.device)

        self.factorized = args.factorized
        if self.factorized:
            self.score = IDQLFactorizedScoreNet(
                latent_dim=self.state_dim,
                action_dim=self.action_dim,
                feature_dim=args.feature_dim,
                embed_dim=args.embed_dim,
                frame_stack=args.frame_stack,
                psi_hidden_depth=args.psi_hidden_depth,
                psi_hidden_dim=args.psi_hidden_dim,
                zeta_hidden_depth=args.zeta_hidden_depth,
                zeta_hidden_dim=args.zeta_hidden_dim,
                score_dropout=args.score_dropout,
                label_dropout=args.label_dropout,
                continuous=False,
            )
        else:
            self.score = IDQLScoreNet(
                latent_dim=self.state_dim,
                action_dim=self.action_dim,
                embed_dim=args.embed_dim,
                hidden_depth=args.zeta_hidden_depth,
                hidden_dim=args.zeta_hidden_dim,
                frame_stack=args.frame_stack,
                score_dropout=args.score_dropout,
                conditional=True,
                continuous=False
            )

    def add_noise(self, x0):
        noise_idx = torch.randint(0, self.sample_steps, (x0.shape[0], )).to(self.device)
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward(self, xt, t, state, action, psi=None):
        if self.factorized:
            if psi is None:
                psi = self.score.forward_psi(state, action)
            score = self.score.forward_score(xt, t, psi=psi)
            return score
        else:
            score = self.score.forward(xt, t, state, action)
            return score

    def compute_loss(self, x0, state, action):
        xt, t, eps = self.add_noise(x0)
        alphabars = self.alphabars[t]
        model_out = self.forward(xt, t.unsqueeze(-1), state, action)
        # loss = (model_out * (1-alphabars).sqrt() + eps).pow(2).sum(-1).mean()
        loss = (model_out - eps).pow(2).sum(-1).mean()  # use eps-prediction for improved performance
        stats = ({
            "info/score_l1_norm": model_out.abs().mean(),
            "info/left_l1_norm": (model_out * (1-alphabars).sqrt()).abs().mean(),
            "info/eps_l1_norm": eps.abs().mean()
        })
        return loss, stats

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
