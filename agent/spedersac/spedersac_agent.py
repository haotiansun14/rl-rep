import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import os

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from utils.util import MLP

from agent.sac.sac_agent import SACAgent
from agent.sac.actor import DiagGaussianActor

from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RFFCritic(nn.Module):

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.outputs = dict()

    def forward(self, critic_feed_feature):
        q1 = torch.sin(self.l1(critic_feed_feature))
        q1 = F.elu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.sin(self.l4(critic_feed_feature))
        q2 = F.elu(self.l5(q2))
        q2 = self.l6(q2)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


class Theta(nn.Module):
    """
    Linear theta
    <phi(s, a), theta> = r
    """

    def __init__(
            self,
            feature_dim=1024,
    ):
        super(Theta, self).__init__()
        self.l = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        r = self.l(feature)
        return r


class SPEDERSACAgent(SACAgent):
    """
    SAC with VAE learned latent features
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            phi_and_mu_lr=-1,
            # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
            phi_hidden_dim=-1,
            phi_hidden_depth=-1,
            mu_hidden_dim=-1,
            mu_hidden_depth=-1,
            critic_and_actor_lr=-1,
            critic_and_actor_hidden_dim=-1,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=1024,
            feature_tau=0.005,
            feature_dim=2048,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )


        self.feature_dim = feature_dim
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps

        self.phi = MLP(input_dim=state_dim + action_dim,
                       output_dim=feature_dim,
                       hidden_dim=phi_hidden_dim,
                       hidden_depth=phi_hidden_depth).to(device)

        if use_feature_target:
            self.phi_target = copy.deepcopy(self.phi)

        self.mu = MLP(input_dim=state_dim,
                      output_dim=feature_dim,
                      hidden_dim=mu_hidden_dim,
                      hidden_depth=mu_hidden_depth).to(device)

        self.theta = Theta(feature_dim=feature_dim).to(device)

        self.feature_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.theta.parameters()),
            weight_decay=0, lr=phi_and_mu_lr)

        self.actor = DiagGaussianActor(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_and_actor_hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5., 2.],
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                weight_decay=0, lr=critic_and_actor_lr,
                                                betas=[0.9, 0.999])  # lower lr for actor/alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_and_actor_lr, betas=[0.9, 0.999])

        self.critic = RFFCritic(feature_dim=feature_dim, hidden_dim=critic_and_actor_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 weight_decay=0, lr=critic_and_actor_lr, betas=[0.9, 0.999])

    def feature_step(self, batch, s_random, a_random, s_prime_random):
        """
        Loss implementation
        """

        state, action, next_state, reward, _ = unpack_batch(batch)

        z_phi = self.phi(torch.concat([state, action], -1))
        z_phi_random = self.phi(torch.concat([s_random, a_random], -1))

        z_mu_next = self.mu(next_state)
        z_mu_next_random = self.mu(s_prime_random)

        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim

        model_loss_pt1 = -2 * z_phi @ z_mu_next.T  # check if need to sum

        model_loss_pt2_a = z_phi_random @ z_mu_next_random.T
        model_loss_pt2 = model_loss_pt2_a @ model_loss_pt2_a.T

        model_loss_pt1_summed = 1. / torch.numel(model_loss_pt1) * torch.sum(model_loss_pt1)
        model_loss_pt2_summed = 1. / torch.numel(model_loss_pt2) * torch.sum(model_loss_pt2)

        model_loss = model_loss_pt1_summed + model_loss_pt2_summed
        r_loss = 0.5 * F.mse_loss(self.theta(z_phi), reward).mean()

        loss = model_loss + r_loss  # + prob_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'total_loss': loss.item(),
            'model_loss': model_loss.item(),
            'r_loss': r_loss.item(),
            # 'prob_loss': prob_loss.item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def critic_step(self, batch):
        """
        Critic update step
        """
        state, action, next_state, reward, done = unpack_batch(batch)

        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)

            z_phi = self.phi(torch.concat([state, action], -1))
            z_phi_next = self.phi(torch.concat([next_state, next_action], -1))

            next_q1, next_q2 = self.critic_target(z_phi_next)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            target_q = reward + (1. - done) * self.discount * next_q

        q1, q2 = self.critic(z_phi)
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item()
        }

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        dist = self.actor(batch.state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        z_phi = self.phi(torch.concat([batch.state, action], -1))

        q1, q2 = self.critic(z_phi)
        q = torch.min(q1, q2)

        actor_loss = ((self.alpha) * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            batch_1 = buffer.sample(batch_size)
            batch_2 = buffer.sample(batch_size)
            s_random, a_random, s_prime_random, _, _ = unpack_batch(batch_2)

            feature_info = self.feature_step(batch_1, s_random, a_random, s_prime_random)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()

        # Critic step
        critic_info = self.critic_step(batch_1)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch_1)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }



