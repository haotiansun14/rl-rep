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
