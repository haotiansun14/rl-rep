from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.lvrep.network import (
    Decoder,
    Encoder,
    GaussianFeature,
    LVRepCritic,
)
from spectralrl.algo.state.sac.agent import SAC
from spectralrl.algo.state.td3.agent import TD3
from spectralrl.module.actor import SquashedDeterministicActor, SquashedGaussianActor
from spectralrl.module.normalize import DummyNormalizer, RunningMeanStdNormalizer
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class LVRep_SAC(SAC):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)
        self.feature_dim = cfg.feature_dim
        self.feature_tau = cfg.feature_tau
        self.use_feature_target = cfg.use_feature_target
        self.feature_update_ratio = cfg.feature_update_ratio
        self.encoder = Encoder(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)
        self.decoder = Decoder(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            hidden_dims=cfg.decoder_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)
        self.f = GaussianFeature(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.f_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)

        if self.use_feature_target:
            self.f_target = make_target(self.f)
        else:
            self.f_target = self.f
        self.feature_optim = torch.optim.Adam(
            [*self.encoder.parameters(), *self.decoder.parameters(), *self.f.parameters()],
            lr=cfg.feature_lr
        )

        # rl networks
        self.actor = SquashedGaussianActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            activation=nn.ELU  # tune
        ).to(self.device)
        self.critic = RFFCritic(
            feature_dim=self.feature_dim,
            num_noise=cfg.num_noise,
            hidden_dim=cfg.critic_hidden_dim,
        ).to(self.device)
        self.critic_target = make_target(self.critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def train(self, training=True):
        super().train(training)
        self.encoder.train(training)
        self.decoder.train(training)
        self.f.train(training)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            feature_loss, feature_metrics = self.feature_step(obs, action, next_obs, reward)
            tot_metrics.update(feature_metrics)
            self.feature_optim.zero_grad()
            feature_loss.backward()
            self.feature_optim.step()

            if self.use_feature_target:
                sync_target(self.f, self.f_target, self.feature_tau)

        critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
        tot_metrics.update(critic_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss, actor_metrics = self.actor_step(obs)
        tot_metrics.update(actor_metrics)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.auto_entropy:
            alpha_loss, alpha_metrics = self.alpha_step(obs)
            tot_metrics.update(alpha_metrics)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        if self._step % self.target_update_freq == 0:
            sync_target(self.critic, self.critic_target, self.tau)

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward):
        z, _, post_mean, post_logstd = self.encoder.sample(obs, action, next_obs, deterministic=False, return_mean_logstd=True)
        s_prime_pred, r_pred = self.decoder.sample(z)
        s_loss = F.mse_loss(s_prime_pred, next_obs)
        r_loss = F.mse_loss(r_pred, reward)
        recon_loss = r_loss + s_loss

        prior_mean, prior_logstd = self.f(obs, action)
        post_var = (2 * post_logstd).exp()
        prior_var = (2 * prior_logstd).exp()
        kl_loss = prior_logstd - post_logstd + 0.5 * (post_var + (post_mean-prior_mean)**2) / prior_var - 0.5
        kl_loss = kl_loss.mean()
        feature_loss = (recon_loss + kl_loss).mean()
        return feature_loss, {
            "loss/feature_loss": feature_loss.item(),
            "loss/kl_loss": kl_loss.item(),
            "loss/recon_loss": recon_loss.item(),
        }

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            next_action, next_logprob, *_ = self.actor.sample(next_obs)
            mean, logstd = self.f_target(obs, action)
            next_mean, next_logstd = self.f_target(next_obs, next_action)
            q_target = self.critic_target(next_mean, next_logstd).min(0)[0] - self.alpha * next_logprob
            q_target = reward + self.discount * (1 - terminal) * q_target
        q_pred = self.critic(mean, logstd)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs):
        new_action, new_logprob, *_ = self.actor.sample(obs)
        mean, logstd = self.f_target(obs, new_action)
        q_value = self.critic(mean, logstd)
        actor_loss = (self.alpha * new_logprob - q_value.min(0)[0]).mean()
        return actor_loss, {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "loss/actor_loss": actor_loss.item()
        }


class LVRep_TD3(TD3):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device,
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)
        self.feature_dim = cfg.feature_dim
        self.feature_tau = cfg.feature_tau
        self.use_feature_target = cfg.use_feature_target
        self.feature_update_ratio = cfg.feature_update_ratio
        self.kl_sep = cfg.kl_sep
        self.kl_coef = cfg.kl_coef
        self.reward_coef = cfg.reward_coef

        self.encoder = Encoder(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)
        self.decoder = Decoder(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            hidden_dims=cfg.decoder_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)
        self.f = GaussianFeature(
            feature_dim=self.feature_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.f_hidden_dims,
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        ).to(self.device)

        if self.use_feature_target:
            self.f_target = make_target(self.f)
        else:
            self.f_target = self.f
        self.feature_optim = torch.optim.Adam(
            [*self.encoder.parameters(), *self.decoder.parameters(), *self.f.parameters()],
            lr=cfg.feature_lr
        )

        # rl networks
        self.actor = SquashedDeterministicActor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm
        ).to(self.device)
        self.critic = LVRepCritic(
            feature_dim=self.feature_dim,
            num_noise=cfg.num_noise,
            hidden_dim=cfg.critic_hidden_dim,
        ).to(self.device)
        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.normalize_obs = cfg.normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdNormalizer(shape=(obs_dim,)).to(self.device)
        else:
            self.obs_rms = DummyNormalizer(shape=(obs_dim,)).to(self.device)

    @torch.no_grad()
    def select_action(self, obs, step=None, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, ...]
        if self.normalize_obs:
            if step is not None:
                # meaning that we are training
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        action, *_ = self.actor.sample(obs, deterministic=deterministic)
        if not deterministic:
            action = action + self.exploration_noise * torch.randn_like(action)
            action = action.clip(-1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy()

    def train(self, training=True):
        super().train(training)
        self.encoder.train(training)
        self.decoder.train(training)
        self.f.train(training)

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)
            feature_loss, feature_metrics = self.feature_step(obs, action, next_obs, reward)
            tot_metrics.update(feature_metrics)
            self.feature_optim.zero_grad()
            feature_loss.backward()
            self.feature_optim.step()

        critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal)
        tot_metrics.update(critic_metrics)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self._step % self.actor_update_freq == 0:
            actor_loss, actor_metrics = self.actor_step(obs)
            tot_metrics.update(actor_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        if self._step % self.target_update_freq == 0:
            self.sync_target()

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward):
        z, _, post_mean, post_logstd = self.encoder.sample(obs, action, next_obs, deterministic=False, return_mean_logstd=True)
        s_prime_pred, r_pred = self.decoder.sample(z)
        s_loss = F.mse_loss(s_prime_pred, next_obs)
        r_loss = F.mse_loss(r_pred, reward)
        recon_loss = r_loss * self.reward_coef + s_loss

        def compute_kl(post_mean, post_logstd, prior_mean, prior_logstd):
            post_var = (2 * post_logstd).exp()
            prior_var = (2 * prior_logstd).exp()
            kl_loss = prior_logstd - post_logstd + 0.5 * (post_var + (post_mean-prior_mean)**2) / prior_var - 0.5
            return kl_loss.mean()

        if self.kl_sep:
            prior_mean_target, prior_logstd_target = self.f_target(obs, action)
            prior_mean, prior_logstd = self.f(obs, action)
            post_kl = compute_kl(post_mean, post_logstd, prior_mean_target, prior_logstd_target)
            prior_kl = compute_kl(post_mean.detach(), post_logstd.detach(), prior_mean, prior_logstd)
            kl_loss = post_kl + prior_kl
        else:
            prior_mean, prior_logstd = self.f(obs, action)
            kl_loss = compute_kl(post_mean, post_logstd, prior_mean, prior_logstd)
        feature_loss = (recon_loss + self.kl_coef * kl_loss)
        return feature_loss, {
            "loss/feature_loss": feature_loss.item(),
            "loss/kl_loss": kl_loss.item(),
            "loss/recon_loss": recon_loss.item(),
            "misc/post_std": post_logstd.exp().mean().item(),
            "misc/prior_std": prior_logstd.exp().mean().item(),
        }

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.sample(next_obs)[0] + noise).clip(-1.0, 1.0)
            mean, logstd = self.f_target(obs, action)
            next_mean, next_logstd = self.f_target(next_obs, next_action)
            q_target = self.critic_target(next_mean, next_logstd).min(0)[0]
            q_target = reward + self.discount * (1 - terminal) * q_target
        q_pred = self.critic(mean, logstd)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs):
        new_action, *_ = self.actor.sample(obs)
        mean, logstd = self.f_target(obs, new_action)
        q_value = self.critic(mean, logstd)
        actor_loss = - q_value.mean()
        return actor_loss, {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "misc/obs_mean": obs.mean().item(),
            "misc/obs_std": obs.std(0).mean().item(),
            "loss/actor_loss": actor_loss.item()
        }

    def sync_target(self):
        super().sync_target()
        sync_target(self.f, self.f_target, self.feature_tau)
