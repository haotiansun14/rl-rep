from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_arch.score_mlp import FourierFeatures, SinusoidalPosEmb
from network_arch.net.mlp import MLP


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
    def __init__(self,
        latent_dim, 
        conditional, 
        action_dim, 
        embed_dim, 
        num_blocks
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.conditional = conditional
        if conditional:
            self.state_embed = nn.Sequential(
                nn.Linear(3*latent_dim, 2*embed_dim), 
                nn.Mish(), 
                nn.Linear(2*embed_dim, 2*embed_dim), 
            )
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim, 2*embed_dim), 
                nn.Mish(), 
                nn.Linear(2*embed_dim, 2*embed_dim), 
            )
            input_dim = latent_dim+5*embed_dim
        else:
            self.state_embed = self.action_embed = None
            input_dim = latent_dim+embed_dim
        # self.time_embed = FourierFeatures(embed_dim)
        self.time_embed = SinusoidalPosEmb(embed_dim)
        self.main = MLPResNet(
            num_blocks=num_blocks, 
            input_dim=input_dim, 
            out_dim=latent_dim, 
            dropout_rate=0.1, 
            use_layer_norm=True, 
            hidden_dim=512, 
            activations=nn.Mish()
        )
        
    def forward(self, next_state_perturbed, state_stack, action, timestep):
        t_ff = self.time_embed(timestep[..., None])
        if self.conditional:
            a_ff = self.action_embed(action)
            B, *_ = state_stack.shape
            s_ff = self.state_embed(state_stack.reshape(B, -1))
            all = torch.concat([next_state_perturbed, s_ff, a_ff, t_ff], dim=-1)
        else:
            all = torch.concat([next_state_perturbed, t_ff], dim=-1)
        return self.main(all)


class IDQLFactoredScoreNet(nn.Module):
    def __init__(
        self, 
        latent_dim: int, 
        action_dim: int, 
        bn_dim: int, 
        feature_dim: int, 
        psi_hidden_dim: int, 
        psi_hidden_depth: int, 
        zeta_hidden_dim: int, 
        zeta_hidden_depth: int
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.time_dim = latent_dim // 2
        if bn_dim is None:
            self.psi_bottleneck1 = self.psi_bottleneck2 = nn.Identity()
            psi_input_dim = latent_dim*3 + action_dim
        else:
            self.psi_bottleneck1 = nn.Sequential(
                nn.Linear(3*latent_dim, bn_dim), 
                nn.LayerNorm(bn_dim), 
                nn.Tanh()
            )
            self.psi_bottleneck2 = nn.Sequential(
                nn.Linear(action_dim, bn_dim), 
                nn.LayerNorm(bn_dim), 
                nn.Tanh()
            )
            psi_input_dim = bn_dim * 2
        self.time_embed = SinusoidalPosEmb(self.time_dim)
        self.psi = MLPResNet(
            num_blocks=psi_hidden_depth, 
            input_dim=psi_input_dim, 
            out_dim=feature_dim, 
            dropout_rate=0.1, 
            use_layer_norm=True, 
            hidden_dim=psi_hidden_dim, 
            activations=nn.Mish()
        )
        self.zeta = MLPResNet(
            num_blocks=zeta_hidden_depth, 
            input_dim=latent_dim+self.time_dim, 
            out_dim=latent_dim*feature_dim, 
            dropout_rate=0.1, 
            use_layer_norm=True, 
            hidden_dim=zeta_hidden_dim, 
            activations=nn.Mish()
        )
        
    def forward_psi(self, state, action):
        s = self.psi_bottleneck1(state.reshape(state.shape[0], -1))
        a = self.psi_bottleneck2(action)
        x = torch.concat([s, a], axis=-1)
        x = self.psi(x)
        return x
    
    def forward_zeta(self, next_state_perturbed, timestep):
        t = self.time_embed(timestep[..., None])
        all = torch.concat([next_state_perturbed, t], dim=-1)
        all = self.zeta(all)
        return all.reshape(-1, self.feature_dim, self.latent_dim)  # note that we did not divide self.feature_dim here

    def forward_score(self, next_state_perturbed, timestep, state=None, action=None, psi=None):
        if psi is None:
            psi = self.forward_psi(state, action)
        score = torch.bmm(
            psi.unsqueeze(1), 
            self.forward_zeta(next_state_perturbed, timestep)
        ).squeeze()
        score /= self.feature_dim   # this is important to scale the range of outputs of both networks
        return score

# class Phi(nn.Module):
#     def __init__(
#         self, 
#         latent_dim: int, 
#         action_dim: int, 
#         bn_dim: Optional[int], 
#         feature_dim: int, 
#         hidden_dim: int, 
#         hidden_depth: int
#     ) -> None:
#         super().__init__()
#         if bn_dim is None:
#             self.bottleneck = nn.Identity()
#             temp_dim = latent_dim*3
#         else:
#             self.bottleneck = nn.Sequential(
#                 nn.Linear(3*latent_dim, bn_dim), 
#                 nn.LayerNorm(bn_dim), 
#                 nn.Tanh()
#             )
#             temp_dim = bn_dim
#         self.model = MLP(
#             input_dim=temp_dim+action_dim, 
#             output_dim=feature_dim, 
#             hidden_dims=[hidden_dim]*hidden_depth
#         )
        
#     def forward(self, current_state, current_action):
#         x = self.bottleneck(current_state.reshape(current_state.shape[0], -1))
#         x = torch.cat([x, current_action], axis=-1)
#         x = self.model(x)
#         return x

# class Phi(nn.Module):
#     def __init__(
#         self,
#         latent_dim: int, 
#         action_dim: int, 
#         bn_dim: Optional[int], 
#         feature_dim: int, 
#         hidden_dim: int, 
#         hidden_depth: int
#     ): 
#         super().__init__()
#         if bn_dim is None:
#             self.bottleneck1 = nn.Identity()
#             self.bottleneck2 = nn.Identity()
#             temp_dim = latent_dim*3+action_dim
#         else:
#             self.bottleneck1 = nn.Sequential(
#                 nn.Linear(3*latent_dim, bn_dim), 
#                 nn.LayerNorm(bn_dim), 
#                 nn.Tanh()
#             )
#             self.bottleneck2 = nn.Sequential(
#                 nn.Linear(action_dim, bn_dim), 
#                 nn.LayerNorm(bn_dim), 
#                 nn.Tanh()
#             )
#             temp_dim = bn_dim*2
        
#         self.model = MLPResNet(
#             num_blocks=hidden_depth, 
#             input_dim=temp_dim, 
#             out_dim=feature_dim, 
#             dropout_rate=0.1, 
#             use_layer_norm=True, 
#             hidden_dim=hidden_dim, 
#             activations=nn.Mish()
#         )
        
#     def forward(self, current_state, current_action):
#         s = self.bottleneck1(current_state.reshape(current_state.shape[0], -1))
#         a = self.bottleneck2(current_action)
#         x = torch.cat([s, a], axis=-1)
#         x = self.model(x)
#         return x


# class NablaMuWithTimeEmbed(nn.Module):
#     def __init__(
#         self, 
#         latent_dim, 
#         action_dim, 
#         embed_dim, 
#         hidden_dim,
#         hidden_depth,  
#         feature_dim
#     ):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.action_dim = action_dim
#         self.feature_dim = feature_dim
#         self.time_embed = SinusoidalPosEmb(embed_dim)
#         self.main = MLPResNet(
#             num_blocks=hidden_depth, 
#             input_dim=latent_dim+embed_dim, 
#             out_dim=latent_dim*feature_dim, 
#             dropout_rate=0.1, 
#             use_layer_norm=True, 
#             hidden_dim=hidden_dim, 
#             activations=nn.Mish()
#         )

#     def forward(self, next_state_perturbed, timestep):
#         t_ff = self.time_embed(timestep[..., None])
#         all = torch.concat([next_state_perturbed, t_ff], dim=-1)
#         return self.main(all).reshape(-1, self.feature_dim, self.latent_dim) / self.feature_dim

        
class RewardDecoder(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        hidden_depth: int, 
        hidden_dim: int
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_depth = hidden_depth
        self.hidden_dim = hidden_dim
        self.ln = nn.LayerNorm(feature_dim)
        self.reward_decoder = MLP(
            input_dim=feature_dim, 
            output_dim=1, 
            hidden_dims=[hidden_dim]*hidden_depth
        )
    
    def forward(self, x):
        x = self.ln(x)
        x = self.reward_decoder(x)
        return x