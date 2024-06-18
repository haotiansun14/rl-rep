import math
import torch
import torch.nn as nn

from helper_functions import util

class Phi(nn.Module):
    # current_state, current_action -> z vector
    def __init__(self, state_dim, action_dim, hidden_dim, feature_dim, hidden_depth):
        super(Phi, self).__init__()

        self.z_vector = util.mlp(state_dim + action_dim, hidden_dim, feature_dim, hidden_depth)


    def forward(self, current_state, current_action):
        x = torch.cat([current_state, current_action], axis=-1)
        z_vector = self.z_vector(x)

        return z_vector


class FeedNablaMu(nn.Module):
    # perturbed_next_state, alpha -> s x z matrix
    def __init__(self, state_dim, hidden_dim, feature_dim, hidden_depth):
        super(FeedNablaMu, self).__init__()


        self.Mu_z_by_s_layer = util.mlp(state_dim + 1, hidden_dim, feature_dim * state_dim, hidden_depth)


    def forward(self, perturbed_next_state, alpha):
        x = torch.cat([perturbed_next_state, alpha], axis=-1)
        score_flat = self.Mu_z_by_s_layer(x)

        return score_flat

        
class NablaMuWithTimeEmbed(nn.Module):
    """
    The network for NablaMu, with sinusoidal timestep embedding. 
    """
    def __init__(
        self, 
        state_dim: int, 
        feature_dim: int, 
        time_dim: int, 
        hidden_dim: int, 
        hidden_depth: int
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim), 
            nn.Linear(time_dim, time_dim*2), 
            nn.ReLU(), 
            nn.Linear(time_dim*2, time_dim)
        )
        self.model = util.mlp(
            state_dim+time_dim, 
            hidden_dim=hidden_dim, 
            output_dim=feature_dim*state_dim, 
            hidden_depth=hidden_depth
        )
    
    def forward(self, perturbed_next_state, timestep):
        o = self.time_embed(timestep)
        o = torch.concat([perturbed_next_state, o])
        return self.model(o)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
