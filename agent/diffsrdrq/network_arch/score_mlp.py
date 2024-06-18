from typing import Optional
import math
import torch
import torch.nn as nn

from helper_functions import util
from network_arch.net.mlp import MLP

class Phi(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        bn_dim: Optional[int], 
        feature_dim: int, 
        hidden_dim: int, 
        hidden_depth: int, 
    ) -> None:
        super(Phi, self).__init__()
        if bn_dim is None:
            self.bottleneck = nn.Identity()
            temp_dim = latent_dim
        else:
            self.bottleneck = nn.Sequential(
                nn.Linear(latent_dim, bn_dim), 
                nn.LayerNorm(bn_dim), 
                nn.Tanh()
            )
            temp_dim = bn_dim
        self.model = MLP(
            input_dim=temp_dim+action_dim, 
            output_dim=feature_dim, 
            hidden_dims=[hidden_dim]*hidden_depth
        )


    def forward(self, current_state, current_action):
        x = self.bottleneck(current_state)
        x = torch.cat([x, current_action], axis=-1)
        x = self.model(x)
        return x
    

class NablaMuWithTimeEmbed(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        time_dim: int, 
        bn_dim: Optional[int], 
        feature_dim: int, 
        hidden_dim: int, 
        hidden_depth: int, 
        embed_type: str="sinusoidal", 
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bn_dim = bn_dim
        if bn_dim is None:
            self.bottleneck = nn.Identity()
            temp_dim = latent_dim
        else:
            self.bottleneck = nn.Sequential(
                nn.Linear(latent_dim, bn_dim), 
                nn.LayerNorm(bn_dim), 
                nn.Tanh()
            )
            temp_dim = bn_dim
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim) if embed_type=="sinusoidal" else FourierFeatures(time_dim), 
            nn.Linear(time_dim, time_dim*2), 
            nn.ReLU(), 
            nn.Linear(time_dim*2, time_dim)
        )
        self.model = MLP(
            input_dim=time_dim+temp_dim, 
            output_dim=temp_dim*feature_dim, 
            hidden_dims=[hidden_dim]*hidden_depth
        )
        if bn_dim is None:
            self.output = nn.Identity()
        else:
            self.output = nn.Linear(bn_dim, latent_dim)
        
    def forward(self, perturbed_next_state, timestep):
        x = torch.concat([
            self.bottleneck(perturbed_next_state), 
            self.time_embed(timestep)
        ], dim=-1)
        x = self.model(x)
        x = x.view(x.shape[0], self.feature_dim, -1)
        return self.output(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FourierFeatures(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn([1, self.dim//2]) * 0.2)
    
    def forward(self, x):
        emb = 2 * math.pi * (x.float() @ self.w)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb