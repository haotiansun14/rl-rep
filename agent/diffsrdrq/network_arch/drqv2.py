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


class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, bn_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, bn_dim),
                                   nn.LayerNorm(bn_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(bn_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(bn_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return torch.stack([q1, q2], dim=0)
        
class Encoder(nn.Module):
    """
    Encoder for three consecutive frames. TODO: consider separately encode
    the frames and concatenate them together? 
    """
    
    def __init__(self, obs_shape):
        super().__init__()
        # self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Flatten()
        )
        self.apply(weight_init)
        with torch.no_grad():
            sample = torch.as_tensor(torch.zeros(obs_shape, dtype=torch.float32)[None, ...]) / 255.0 - 0.5
            self.repr_dim = self.convnet(sample).shape[1]

    def forward(self, obs):
        obs = obs / 255.0 - 0.5 # normalize to [-0.5, 0.5]
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    

class Decoder(nn.Module):
    def __init__(self, obs_shape = (32,35,35)):
        super().__init__()
        # self.repr_dim = 32 * 35 * 35
        self.deconvnet = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 37, 37])
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 39, 39])
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 41, 41])
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, 3, stride=2), # torch.Size([256, 32, 83, 83])
            nn.ReLU(), 
            nn.Conv2d(32, 9, 2, stride=1,padding=1), # torch.Size([256, 3, 84, 84])
        )
        self.apply(weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], 32, 35, 35)
        h = self.deconvnet(obs)
        return h

        
class PredictEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        # (size - (kernel_size - 1) - 1) // stride + 1

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5 # normalize to [-0.5, 0.5]
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h