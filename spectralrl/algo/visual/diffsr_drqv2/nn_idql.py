import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

    def forward(self, input: Tensor) -> Tensor:
        f = 2 * math.pi * input @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else None

    def forward(self, x, training=False):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim, self.hidden_dim)

        self.blocks = nn.ModuleList([MLPResNetBlock(self.hidden_dim, self.activations, self.dropout_rate, self.use_layer_norm)
                                     for _ in range(self.num_blocks)])

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x


class IDQLScoreNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        embed_dim: int,
        hidden_depth: int,
        hidden_dim: int,
        frame_stack: int,
        score_dropout: Optional[float]=None,
        conditional: bool=True,
        continuous: bool=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.conditional = conditional
        self.time_embed = FourierFeature(embed_dim) if continuous else PositionalFeature(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        if conditional:
            self.state_mlp = nn.Sequential(
                nn.Linear(latent_dim*frame_stack, embed_dim*2),
                nn.Mish(),
                nn.Linear(embed_dim*2, embed_dim)
            )
            self.action_mlp = nn.Sequential(
                nn.Linear(action_dim, embed_dim*2),
                nn.Mish(),
                nn.Linear(embed_dim*2, embed_dim)
            )
            input_dim = latent_dim + embed_dim*3
        else:
            self.state_mlp = self.action_mlp = None
            input_dim = latent_dim
        self.main = MLPResNet(
            num_blocks=hidden_depth,
            input_dim=input_dim,
            out_dim=latent_dim,
            dropout_rate=score_dropout,
            use_layer_norm=True,
            hidden_dim=hidden_dim,
            activations=nn.Mish()
        )

    def forward(self, next_state_perturbed, timestep, state, action):
        t_ff = self.time_embed(timestep)
        t_ff = self.time_mlp(t_ff)
        if self.conditional:
            a_ff = self.action_mlp(action)
            s_ff = self.state_mlp(state)
            all = torch.concat([next_state_perturbed, s_ff, a_ff, t_ff], dim=-1)
        else:
            all = torch.concat([next_state_perturbed, t_ff], dim=-1)
        return self.main(all)


class IDQLFactorizedScoreNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        feature_dim: int,
        embed_dim: int,
        frame_stack: int,
        psi_hidden_depth: int,
        psi_hidden_dim: int,
        zeta_hidden_depth: int,
        zeta_hidden_dim: int,
        score_dropout: Optional[float]=None,
        label_dropout: Optional[float]=None,
        continuous: bool=False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(latent_dim*frame_stack, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.time_embed = FourierFeature(embed_dim) if continuous else PositionalFeature(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.Mish(),
            nn.Linear(embed_dim*2, embed_dim)
        )

        # psi and zeta networks
        self.psi = MLPResNet(
            num_blocks=psi_hidden_depth,
            input_dim=embed_dim*2,
            out_dim=feature_dim,
            dropout_rate=score_dropout,
            use_layer_norm=True,
            hidden_dim=psi_hidden_dim,
            activations=nn.Mish()
        )
        self.zeta = MLPResNet(
            num_blocks=zeta_hidden_depth,
            input_dim=latent_dim+embed_dim,
            out_dim=latent_dim*feature_dim,
            dropout_rate=score_dropout,
            use_layer_norm=True,
            hidden_dim=zeta_hidden_dim,
            activations=nn.Mish()
        )
        self.label_dropout = label_dropout
        if self.label_dropout:
            self.psi_act = nn.Softmax(dim=-1)
        else:
            self.psi_act = nn.Identity()

    def forward_psi(self, state, action):
        rand = torch.rand([*state.shape[:-1], 1], device=state.device)
        # non-dropout
        s_ff = self.state_mlp(state)
        a_ff = self.action_mlp(action)
        x = torch.concat([s_ff, a_ff], dim=-1)
        x = self.psi(x)
        x_nondrop = self.psi_act(x)

        if self.label_dropout and self.training:
            # label dropout
            x_drop = torch.ones_like(x_nondrop) / self.feature_dim
            x = torch.where(rand < self.label_dropout, x_drop, x_nondrop)
        else:
            x = x_nondrop

        return x

    def forward_zeta(self, next_state_perturbed, timestep):
        t_ff = self.time_embed(timestep)
        t_ff = self.time_mlp(t_ff)
        all = torch.concat([next_state_perturbed, t_ff], dim=-1)
        all = self.zeta(all)
        return all.reshape(-1, self.feature_dim, self.latent_dim)  # note that we did not divide self.feature_dim here

    def forward_score(self, next_state_perturbed, timestep, state=None, action=None, psi=None):
        if psi is None:
            psi = self.forward_psi(state, action)
        score = torch.bmm(
            psi.unsqueeze(1),
            self.forward_zeta(next_state_perturbed, timestep)
        ).squeeze()
        # score /= self.feature_dim  # check whether this is needed
        return score
