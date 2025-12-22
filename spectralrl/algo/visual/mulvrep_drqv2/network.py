import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from spectralrl.algo.visual.drqv2.network import (
    Actor,
    Encoder,
    RandomShiftsAug,
    setup_schedule,
    weight_init,
)


class Decoder(nn.Module):
    def __init__(self, obs_shape = (32,35,35)):

        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.deconvnet = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 37, 37])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 39, 39])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 41, 41])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=2), # torch.Size([256, 32, 83, 83])
                                     nn.ReLU(), nn.Conv2d(32, 3, 2, stride=1,padding=1), # torch.Size([256, 3, 84, 84])
                                    )
        self.apply(weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], 32, 35, 35)
        h = self.deconvnet(obs) # [0,+inf] because of ReLU
        return h


class PredictEncoder(nn.Module):
    # encoder for the predicted image
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


class RFFCritic(nn.Module):
    """
    Critic with random fourier features
    """

    def __init__(
            self,
            feature_dim,
            c_noise,
            q_activ,
            num_noise=20,
            hidden_dim=256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_noise = num_noise
        self.c_noise = c_noise
        if q_activ == 'relu':
            self.q_activ = F.relu
        elif q_activ == 'elu':
            self.q_activ = F.elu
        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, mean, log_std):
        std = log_std.exp()
        batch_size, d = mean.shape

        x = mean[:, None, :] + std[:, None, :] * torch.randn([self.num_noise, self.feature_dim], requires_grad=False, device=mean.device) * self.c_noise
        x = x.reshape(-1, d)

        q1 = self.q_activ(self.l1(x))  # F.relu(self.l1(x))
        q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = self.q_activ(self.l2(q1))  # F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self.q_activ(self.l4(x))  # F.relu(self.l4(x))
        q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = self.q_activ(self.l5(q2))  # F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

class FeatEncoder(nn.Module):
    """
    Gaussian encoder

    s,a,s' -> z
    """
    def __init__(
        self,
        repr_dim,
        action_dim,
        feature_dim=256,
    ):

        super(FeatEncoder, self).__init__()

        input_dim = repr_dim + action_dim + repr_dim
        self.mean_linear = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.log_std_linear = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )


    def forward(self, state, action, next_state):
        x = torch.cat([state, action, next_state], axis=-1)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state, action, next_state):
        mean, log_std = self.forward(state, action, next_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample() # reparameterization
        return z


class FeatDecoder(nn.Module):
    """
    Deterministic decoder (Gaussian with identify covariance)

    z -> s
    """
    def __init__(
        self,
        repr_dim,
        feature_dim=256,
        hidden_dim=256,
    ):

        super(FeatDecoder, self).__init__()

        self.l1 = nn.Linear(feature_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.state_linear = nn.Linear(hidden_dim, repr_dim)
        self.reward_linear = nn.Linear(hidden_dim, 1)


    def forward(self, feature):
        """
        Decode an input feature to observation
        """
        x = F.relu(self.l1(feature)) #F.relu(self.l1(feature))
        x = F.relu(self.l2(x))
        s = self.state_linear(x)
        r = self.reward_linear(x)
        return s, r


class GaussianFeature(nn.Module):
    """
    Gaussian feature extraction with parameterized mean and std

    s,a -> z
    """
    def __init__(
        self,
        repr_dim,
        action_dim,
        feature_dim=256,
    ):


        super(GaussianFeature, self).__init__()
        self.mean_linear = nn.Sequential(
            nn.Linear(repr_dim + action_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.log_std_linear = nn.Sequential(
            nn.Linear(repr_dim + action_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], axis=-1)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std
