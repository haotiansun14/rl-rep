import torch
import torch.nn as nn
from network_arch.net.mlp import EnsembleMLP


def get_critic(critic_version, *args, **kwargs):
    if critic_version == "prf": 
        return PRFCritic(*args, **kwargs)
    elif critic_version == "control": 
        return ControlCritic(*args, **kwargs)
    elif critic_version == "rff": 
        return RFFCritic(*args, **kwargs)
    elif critic_version == 'rff_reg':
        return RFFRegCritic(*args, **kwargs)
    else:
        raise NotImplementedError


class Sin(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.sin(x)
    
class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.exp(x)
    

class PRFCritic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int=256,
    ):
        super().__init__()
        self.model = EnsembleMLP(
            input_dim=input_dim, 
            output_dim=1, 
            ensemble_size=2, 
            hidden_dims=[hidden_dim, hidden_dim], 
            activation=[Exp, nn.ELU()]
        )
        
    def forward(self, x):
        return self.model(x)


class RFFCritic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int=256
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.model = EnsembleMLP(
            input_dim=input_dim, 
            output_dim=1, 
            ensemble_size=2, 
            hidden_dims=[hidden_dim, hidden_dim], 
            activation=[Sin, nn.ELU]
        )
        
    def forward(self, x):
        x = self.ln(x)
        return self.model(x)
    
class RFFRegCritic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int=256, 
    ):
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


class ControlCritic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int=256
    ):
        super().__init__()
        self.model = EnsembleMLP(
            input_dim=input_dim, 
            output_dim=1, 
            ensemble_size=2, 
            hidden_dims=[hidden_dim, hidden_dim]
        )
        
    def forward(self, x):
        return self.model(x)