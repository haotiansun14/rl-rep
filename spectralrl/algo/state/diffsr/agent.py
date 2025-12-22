from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralrl.algo.state.diffsr.ddpm import DDPM
from spectralrl.algo.state.diffsr.network import RFFCritic
from spectralrl.algo.state.diffsr.vae import VAE
from spectralrl.algo.state.td3.agent import TD3
from spectralrl.module.actor import SquashedDeterministicActor, SquashedGaussianActor
from spectralrl.module.normalize import DummyNormalizer, RunningMeanStdNormalizer
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class DiffSR_TD3(TD3):
    def __init__(
        self,
        obs_dim,
        action_dim,
        cfg,
        device
    ) -> None:
        super().__init__(obs_dim, action_dim, cfg, device)
        self.feature_dim = cfg.feature_dim
        self.feature_update_ratio = cfg.feature_update_ratio
        self.back_critic_grad = cfg.back_critic_grad
        self.recon_coef = cfg.recon_coef
        self.kl_coef = cfg.kl_coef
        self.diffusion_coef = cfg.diffusion_coef
        self.critic_coef = cfg.critic_coef
        self.reward_coef = cfg.reward_coef
        self.use_latent = cfg.use_latent
        self.normalize_obs = cfg.normalize_obs

        # vae
        if self.use_latent:
            self.vae = VAE(
                cfg,
                obs_dim,
                device=self.device
            ).to(self.device)
            self.vae_target = make_target(self.vae)
            self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=cfg.vae_lr)
            actual_obs_dim = cfg.latent_dim
        else:
            actual_obs_dim = obs_dim

        # diffusion
        self.diffusion = DDPM(
            cfg,
            actual_obs_dim,
            action_dim,
            device=self.device
        ).to(self.device)
        self.diffusion_target = make_target(self.diffusion)
        self.diffusion_optim = torch.optim.Adam(self.diffusion.parameters(), lr=cfg.diffusion_lr)

        # rl network
        self.actor = SquashedDeterministicActor(
            input_dim=actual_obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm
        ).to(self.device)
        self.critic = RFFCritic(
            feature_dim=self.feature_dim,
            hidden_dim=cfg.critic_hidden_dim
        ).to(self.device)
        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        if self.normalize_obs:
            self.obs_rms = RunningMeanStdNormalizer(shape=(obs_dim,)).to(self.device)
        else:
            self.obs_rms = DummyNormalizer(shape=(obs_dim,)).to(self.device)

    def train(self, training):
        super().train(training)
        if self.use_latent:
            self.vae.train(training)
        self.diffusion.train(training)

    @torch.no_grad()
    def select_action(self, obs, step=None, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, ...]
        if self.normalize_obs:
            if step is not None:
                # meaning that we are training
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)
        if self.use_latent:
            latent, _ = self.vae(obs, sample_posterior=False, forward_decoder=False)
        else:
            latent = obs
        action, *_ = self.actor.sample(latent, deterministic=deterministic)
        if not deterministic:
            action = action + self.exploration_noise * torch.randn_like(action)
            action = action.clip(-1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy()

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device) for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]
            obs = self.obs_rms.normalize(obs)
            next_obs = self.obs_rms.normalize(next_obs)
            recon_loss, kl_loss, diffusion_loss, reward_loss, metrics, latent, next_latent = \
                self.feature_step(obs, action, next_obs, reward)
            tot_metrics.update(metrics)

            critic_loss, critic_metrics = self.critic_step(obs, action, next_obs, reward, terminal, latent=latent)
            tot_metrics.update(critic_metrics)

            if self.use_latent:
                self.vae_optim.zero_grad()
            self.diffusion_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss = (
                recon_loss * self.recon_coef + \
                kl_loss * self.kl_coef + \
                diffusion_loss * self.diffusion_coef + \
                reward_loss * self.reward_coef + \
                critic_loss * self.critic_coef
            )
            loss.backward()
            if self.use_latent:
                self.vae_optim.step()
            self.diffusion_optim.step()
            self.critic_optim.step()

        if self._step % self.actor_update_freq == 0:
            actor_loss, actor_metrics = self.actor_step(obs, latent=latent)
            tot_metrics.update(actor_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        if self._step % self.target_update_freq == 0:
            self.sync_target()

        self._step += 1
        return tot_metrics

    def feature_step(self, obs, action, next_obs, reward):
        if self.use_latent:
            B = obs.shape[0]
            all_obs = torch.concat([obs, next_obs], dim=0)
            _, all_latent_dist, all_obs_pred = self.vae(all_obs, sample_posterior=True, forward_decoder=True)
            recon_loss = F.mse_loss(all_obs_pred[:B], obs, reduction="sum") / B
            kl_loss = all_latent_dist.kl()[:B].mean()
            latent, next_latent = torch.split(all_latent_dist.mode(), [B, B], dim=0)
        else:
            recon_loss = kl_loss = torch.tensor(0.0)
            latent, next_latent = obs, next_obs

        diffusion_loss, reward_loss, stats = self.diffusion.compute_loss(
            next_latent,
            latent,
            action,
            reward
        )
        metrics = {
            "loss/recon_loss": recon_loss.item(),
            "loss/kl_loss": kl_loss.item(),
            "loss/diffusion_loss": diffusion_loss.item(),
            "loss/reward_loss": reward_loss.item()
        }
        metrics.update(stats)
        return recon_loss, kl_loss, diffusion_loss, reward_loss, metrics, latent, next_latent

    def critic_step(self, obs, action, next_obs, reward, terminal, latent):
        if self.use_latent:
            latent = latent
            next_latent, *_ = self.vae_target(next_obs, sample_posterior=False, forward_decoder=False)
        else:
            latent = obs
            next_latent = next_obs
        if self.back_critic_grad:
            feature = self.diffusion.forward_psi(s=latent, a=action)
        else:
            feature = self.diffusion_target.forward_psi(s=latent, a=action)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target.sample(next_latent)[0] + noise).clip(-1.0, 1.0)
            next_feature = self.diffusion_target.forward_psi(s=next_latent, a=next_action)
            q_target = self.critic_target(next_feature).min(0)[0]
            q_target = reward + self.discount * (1 - terminal) * q_target
        q_pred = self.critic(feature)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss.item(),
        }

    def actor_step(self, obs, latent):
        if self.use_latent:
            latent = latent.detach()
        else:
            latent = obs
        new_action, *_ = self.actor.sample(latent)
        if self.back_critic_grad: # should be consistent to critic training
            new_feature = self.diffusion.forward_psi(s=latent, a=new_action)
        else:
            new_feature = self.diffusion_target.forward_psi(s=latent, a=new_action)
        q_value = self.critic(new_feature)
        actor_loss =  - q_value.mean()
        return actor_loss, {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "loss/actor_loss": actor_loss.item()
        }

    def sync_target(self):
        super().sync_target()
        sync_target(self.diffusion, self.diffusion_target, self.tau)
        if self.use_latent:
            sync_target(self.vae, self.vae_target, self.tau)
