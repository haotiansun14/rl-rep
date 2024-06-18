import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from network_arch.actor import SquashedNormal
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            
            
class RandomShiftsAug(nn.Module):
    """
    Random Shift Augmentation for input images
    """
    def __init__(self, pad: int) -> None:
        super().__init__()
        self.pad = pad
        
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(
            x,
            grid,
            padding_mode='zeros',
            align_corners=False
        )
        
class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class Actor(nn.Module):
    """Basically this is the same with DrQ-V2 actor"""
    def __init__(self, repr_dim, action_dim, bn_dim, hidden_dim):
        super().__init__()

        if bn_dim is not None:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, bn_dim),
                                    nn.LayerNorm(bn_dim), nn.Tanh())
            input_dim = bn_dim
        else:
            self.trunk = nn.Identity()
            input_dim = repr_dim

        self.policy = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class RFFCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(input_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, return_feature=False):
        q1_feature, q2_feature = self.forward_feature(x)
        q1 = self.l3(q1_feature)
        q2 = self.l6(q2_feature)
        if return_feature:
            return torch.stack([q1, q2], dim=0), q1_feature, q2_feature
        else:
            return torch.stack([q1, q2], dim=0)

    def forward_feature(self, x):
        x = self.ln(x)
        q1 = torch.sin(self.l1(x))
        q1 = torch.nn.functional.elu(self.l2(q1))

        q2 = torch.sin(self.l4(x))
        q2 = torch.nn.functional.elu(self.l5(q2))
        return q1, q2