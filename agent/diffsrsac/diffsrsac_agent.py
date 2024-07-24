import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

from agent.sac.sac_agent import SACAgent
from scipy.stats import beta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class RFFCritic(nn.Module):

    def __init__(self, critic_feed_feature_dim,
                 hidden_dim,
                 critic_elu_layer_regularizer_lambda):
        super().__init__()

        self.critic_elu_layer_regularizer_lambda = critic_elu_layer_regularizer_lambda

        # Q1
        self.l1 = nn.Linear(critic_feed_feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(critic_feed_feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.outputs = dict()

    def forward(self, critic_feed_feature):
        def get_reg_term(batch_feature, regularizer_lambda):
            c = 1.
            x = batch_feature

            n = x.shape[0]
            d = x.shape[1]
            inprods = x @ x.T
            norms = inprods[torch.arange(n), torch.arange(n)]
            part1 = inprods.pow(2).sum() - norms.pow(2).sum()
            part1 = part1 / ((n - 1) * n)
            part2 = - 2 * c * norms.mean() / d
            part3 = c * c / d

            return regularizer_lambda * (part1 + part2 + part3)

        q1 = torch.sin(self.l1(critic_feed_feature))
        q1 = F.elu(self.l2(q1))
        reg_term_q1_elu = get_reg_term(self.l2(q1), regularizer_lambda=self.critic_elu_layer_regularizer_lambda)
        q1 = self.l3(q1)

        q2 = torch.sin(self.l4(critic_feed_feature))
        q2 = F.elu(self.l5(q2))
        reg_term_q2_elu = get_reg_term(self.l5(q2), regularizer_lambda=self.critic_elu_layer_regularizer_lambda)
        q2 = self.l6(q2)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2, reg_term_q1_elu + reg_term_q2_elu


class DIFFSRSACAgent(SACAgent):

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            feature_dim=256,
            phi_and_nabla_mu_lr=0.003,
            phi_hidden_dim=256,
            phi_hidden_depth=1,
            nabla_mu_hidden_dim=512,
            nabla_mu_hidden_depth=1,
            critic_and_actor_lr=3e-4,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            extra_feature_steps=3,
            num_noises=1000,
            critic_elu_layer_regularizer_lambda=0,
            DARL_noise_a=0.3,
            DARL_noise_b=0.1,
            sigma_scale_factor=0.449
    ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=critic_and_actor_lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
        )

        self.num_noises = num_noises
        self.noise_alphabars, self.noise_alphas = self.generate_alphabars_and_alphas(a=DARL_noise_a, b=DARL_noise_b,
                                                                                     num_alphas=(self.num_noises))
        self.noise_alphabars, self.noise_alphas = self.noise_alphabars.to(device), self.noise_alphas.to(device)
        self.sigma_scale_factor = sigma_scale_factor

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.phi_hidden_dim = phi_hidden_dim
        self.phi_hidden_depth = phi_hidden_depth
        self.nabla_mu_hidden_dim = nabla_mu_hidden_dim
        self.nabla_mu_hidden_depth = nabla_mu_hidden_depth

        self.extra_feature_steps = extra_feature_steps

        self.critic_feed_feature = Phi(state_dim=state_dim, action_dim=action_dim, hidden_dim=phi_hidden_dim,
                                       feature_dim=feature_dim, hidden_depth=phi_hidden_depth).to(device)

        self.nablamu_net = FeedNablaMu(feature_dim=feature_dim, state_dim=state_dim,
                                       hidden_dim=nabla_mu_hidden_dim, hidden_depth=nabla_mu_hidden_depth).to(device)

        self.critic = RFFCritic(
            critic_feed_feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            critic_elu_layer_regularizer_lambda=critic_elu_layer_regularizer_lambda

        ).to(device)
        self.critic_target = RFFCritic(
            critic_feed_feature_dim=self.feature_dim,
            hidden_dim=hidden_dim,
            critic_elu_layer_regularizer_lambda=critic_elu_layer_regularizer_lambda

        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.phi_optimizer = torch.optim.Adam(list(self.critic_feed_feature.parameters()),
                                              lr=phi_and_nabla_mu_lr,
                                              betas=[0.9, 0.999])

        self.nablamu_net_optimizer = torch.optim.Adam(list(self.nablamu_net.parameters()),
                                                      lr=phi_and_nabla_mu_lr,
                                                      betas=[0.9, 0.999])

    @staticmethod
    def generate_alphabars_and_alphas(a, b, num_alphas):

        def to_tensor(arr):
            # convert arr to tensor
            return torch.tensor(arr).float()

        def get_array_from_cumulative_prod_of_array_sota(alphabars):
            def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.99):
                betas = []
                for i in range(num_diffusion_timesteps):
                    t1 = i  # / num_diffusion_timesteps
                    t2 = (i + 1)  # / num_diffusion_timesteps
                    betas.append(min(1 - alpha_bar[t2] / alpha_bar[t1], max_beta))
                return np.concatenate([np.array([betas[0]]), np.array(betas)])

            return betas_for_alpha_bar(len(alphabars) - 1, alphabars)

        x = np.linspace(0, 1, num_alphas)  # Define x-axis values for the distribution
        cdf = beta.cdf(x, a, b)

        raw_vals = 1. - cdf
        alphabars_custom = np.clip(raw_vals, a_min=raw_vals[-2], a_max=raw_vals[1])
        alphas_custom = 1. - get_array_from_cumulative_prod_of_array_sota(raw_vals)

        return to_tensor(alphabars_custom), to_tensor(alphas_custom)

    def critic_step(self, batch):
        """
        Critic update step
        """
        obs, action, next_obs, reward, done = util.unpack_batch(batch)
        not_done = 1. - done

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2, reg_term_target = self.critic_target(self.critic_feed_feature(next_obs, next_action))
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2, reg_term_current = self.critic(self.critic_feed_feature(obs, action))

        critic_loss_noreg = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        critic_loss = critic_loss_noreg + reg_term_target + reg_term_current

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'q_loss_reg': critic_loss.item(),
            'q_loss_noreg': critic_loss_noreg.item(),
            'q1': current_Q1.mean().item(),
            'q2': current_Q1.mean().item()
        }

    def update_actor_and_alpha(self, batch):
        obs = batch.state

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2, _ = self.critic(self.critic_feed_feature(obs, action))

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss
            info['alpha'] = self.alpha

        return info

    def critic_feeder_feature_step(self, batch):

        # selecting noise alphas
        batch_size = batch.action.shape[0]
        current_state_batch, current_action_batch, next_state_batch = batch.state, batch.action, batch.next_state
        noise_alphas_random_idx = torch.randint(0, self.num_noises, (batch_size,)).to(device)
        selected_noise_alphas = torch.index_select(self.noise_alphabars, 0, noise_alphas_random_idx).float().to(
            device)
        selected_noise_alphas = selected_noise_alphas.reshape((selected_noise_alphas.shape[0], 1))

        # sampling random noise
        sigma_squared = torch.normal(mean=torch.zeros_like(next_state_batch),
                                     std=torch.ones_like(next_state_batch) * self.sigma_scale_factor).to(device)

        # perturbing next state
        next_state_batch_perturbed = (torch.sqrt(selected_noise_alphas) * next_state_batch + torch.sqrt(
            1.0 - selected_noise_alphas) * sigma_squared)
        next_state_tilde_minus_alpha_next_state = -(
                next_state_batch_perturbed - torch.sqrt(selected_noise_alphas) * next_state_batch)

        # phi out, \phi(state, action) -> latent
        phi_out_dim_1_by_z = self.critic_feed_feature(current_state_batch, current_action_batch)

        # nabla_mu out, \nabla_mu(next_state_perturbed, \alpha) -> latent by state
        nabla_mu_out_dim_z_by_s_flat = self.nablamu_net(next_state_batch_perturbed, selected_noise_alphas)
        nabla_mu_out_dim_z_by_s = nabla_mu_out_dim_z_by_s_flat.reshape((batch_size, self.feature_dim, self.state_dim))

        # batch multiply phi and nabla_mu
        score = torch.bmm(phi_out_dim_1_by_z.unsqueeze(1), nabla_mu_out_dim_z_by_s).squeeze()
        grad_log_prob_perturbed_s = (1. - selected_noise_alphas) * self.sigma_scale_factor * score
        diff = next_state_tilde_minus_alpha_next_state - grad_log_prob_perturbed_s

        # calculate score loss
        score_loss_batch = (1 / batch_size) * torch.sum(diff ** 2, dim=[i for i in range(1, len(diff.shape))])
        score_loss = score_loss_batch.sum()

        # step through neural networks
        self.nablamu_net_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()

        score_loss.backward()

        self.nablamu_net_optimizer.step()
        self.phi_optimizer.step()

        return {
            'score_loss': score_loss.item(),
        }

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        for _ in range(self.extra_feature_steps+1):
            batch = buffer.sample(batch_size)
            feature_info = self.critic_feeder_feature_step(batch)

        # A critic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }
