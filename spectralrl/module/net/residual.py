from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float=0.0,
        activation: nn.Module=nn.ReLU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = activation
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x + self.residual(residual)

class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int=0,
        hidden_dims: Sequence[int]=[],
        activation: Callable=F.relu,
        dropout: float=None,
        device: str="cpu",
    ):
        super().__init__()

        assert len(hidden_dims) > 0, "hidden_dims must be a list of at least one integer"
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        hidden_dims = [hidden_dims[0]] + list(hidden_dims)
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(
                hidden_dims[i],
                hidden_dims[i+1],
                dropout,
                activation,
            ) for i in range(len(hidden_dims) - 1)
        ])

    def forward(self, x):
        x = self.fc1(x)


    def forward(self, x):
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
