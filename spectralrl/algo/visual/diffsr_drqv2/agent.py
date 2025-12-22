from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spectralrl.algo.visual.base import BaseVisualAlgorithm
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target

from .ddpm import DDPM
from .network import Actor, RandomShiftsAug, RFFCritic, setup_schedule
from .nn_vae import VAE1D, Scaler


class DiffSR_DrQv2(BaseVisualAlgorithm):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        device
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape[0]
        self.use_repr_target = cfg.use_repr_target
        self.kl_coef = cfg.kl_coef
        self.recon_coef = cfg.recon_coef
        self.diffusion_coef = cfg.diffusion_coef
        self.critic_coef = cfg.critic_coef
        self.reg_coef = cfg.reg_coef
        self.tau = cfg.tau

        self.extra_repr_step = cfg.extra_repr_step
        self.update_every = cfg.update_every

        self.stddev_schedule = setup_schedule(cfg.stddev_schedule)
        self.stddev_clip = cfg.stddev_clip
        self.sample_steps = cfg.sample_steps

        self.ae_pretrain_steps = cfg.ae_pretrain_steps
        self.back_critic_grad = cfg.back_critic_grad
        self.critic_loss_type = cfg.critic_loss_type
        if self.critic_loss_type == "mse":
            self.critic_loss_fn = F.mse_loss
        elif self.critic_loss_type == "huber":
            self.critic_loss_fn = partial(F.huber_loss, delta=250.0)

        # networks
        self.vae = VAE1D(cfg).to(self.device)
        self.scaler = Scaler(activate=cfg.do_scale)
        self.diffusion = DDPM(
            cfg,
            state_dim=cfg.latent_dim,
            action_dim=self.action_dim,
            device=device
        ).to(self.device)
        self.actor = Actor(
            repr_dim=cfg.latent_dim*3,
            action_dim=self.action_dim,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.actor_hidden_dim
        ).to(self.device)
        self.critic = RFFCritic(
            input_dim=cfg.feature_dim,
            hidden_dim=cfg.critic_hidden_dim
        ).to(self.device)
        self.critic_target = make_target(self.critic)
        if self.use_repr_target:
            self.vae_target = make_target(self.vae)
            self.diffusion_target = make_target(self.diffusion)
        else:
            self.vae_target = self.vae
            self.diffusion_target = self.diffusion
        self.aug = RandomShiftsAug(pad=4)

        # optimizers
        self.optim = {
            "vae": torch.optim.Adam(self.vae.parameters(), lr=cfg.ae_lr),
            "diffusion": torch.optim.AdamW(self.diffusion.parameters(), lr=cfg.diffusion_lr),
            "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr),
        }
        self._step = 1
        self.train()

    def train(self, training=True):
        self.training = training
        for module in [self.vae, self.diffusion, self.actor, self.critic]:
            module.train(training)

    @torch.no_grad()
    def evaluate(self, replay_iter):
        self.train(False)
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = img_stack.float()
        next_img_step = next_img_step[:, -3:, ...].float()

        vae, diffusion = self.vae_target, self.diffusion_target

        B, S, W, H = img_stack.shape
        img_stack = img_stack.view(B*3, S//3, W, H)
        latent_stack, _ = vae(img_stack, sample_posterior=False, forward_decoder=False)
        latent_stack = latent_stack.reshape(B, -1)
        next_latent, _ = vae(next_img_step, sample_posterior=False, forward_decoder=False)
        raw_next_img = next_img_step[:9, :] / 255. - 0.5
        recon_next_img = vae.decode(next_latent)[:9, :]

        # DDPM sampling
        latent_stack = self.scaler(latent_stack)
        next_latent = self.scaler(next_latent)

        with torch.no_grad():
            _, info = diffusion.sample(next_latent.shape, latent_stack, action, preserve_history=True)
            history = info["sample_history"]

        stepsize = diffusion.sample_steps // 5
        history = list(reversed(history))
        checkpoints = list(range(0, self.sample_steps, stepsize)) + [self.sample_steps]
        checkpoints = list(reversed(checkpoints))
        checkpoint_next_latents = [history[c] for c in checkpoints]
        checkpoint_gen_next_imgs = [vae.decode(xt)[:9, :] for xt in checkpoint_next_latents]

        checkpoint_next_latents = torch.stack(checkpoint_next_latents, dim=0)
        img_to_show = torch.stack([raw_next_img, recon_next_img]+checkpoint_gen_next_imgs, dim=0)
        latent_l1_diff = (checkpoint_next_latents - next_latent.repeat(6, 1, 1)).abs().mean([1,2])
        N = img_to_show.shape[0]
        img_to_show = img_to_show.reshape(N*9, 3, img_to_show.shape[-2], img_to_show.shape[-1])
        img_to_show = img_to_show + 0.5
        grid = torchvision.utils.make_grid(img_to_show, nrow=9)

        metrics = {
            f"l1diff_step{checkpoints[i]}": latent_l1_diff[i].item() for i in range(len(checkpoints))
        }
        self.train(True)
        torch.cuda.empty_cache()
        return metrics, grid

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)[None, ...]
        obs = obs.view(3, 3, obs.shape[-1], obs.shape[-1])
        obs, _ = self.vae_target(obs, sample_posterior=False, forward_decoder=False)
        obs = self.scaler.forward(obs)
        obs = obs.reshape(1, -1)
        stddev = self.stddev_schedule(step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=None) if not deterministic else dist.mean
        return action.squeeze().cpu().numpy()

    def update_target(self, tau):
        sync_target(self.critic, self.critic_target, tau)
        if self.use_repr_target:
            sync_target(self.vae, self.vae_target, tau)
            sync_target(self.diffusion, self.diffusion_target, tau)

    def pretrain_step(self, replay_iter, step):
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = self.aug(img_stack.float()).detach()
        next_img_step = next_img_step[:, -3:].float().detach()
        tot_metrics = {}
        if step < self.ae_pretrain_steps:
            ae_metrics, recon_loss, kl_loss, latent, next_latent_step, latent_mode, next_latent_step_mode = self.ae_step(img_stack, next_img_step)
            tot_metrics.update(ae_metrics)
            self.optim["vae"].zero_grad(set_to_none=True)
            (recon_loss*self.recon_coef + kl_loss*self.kl_coef).backward()
            self.optim["vae"].step()
        else:
            ae_metrics, recon_loss, kl_loss, latent, next_latent_step, latent_mode, next_latent_step_mode = self.ae_step(img_stack, next_img_step)
            tot_metrics.update(ae_metrics)
            score_metrics, diffusion_loss, reg_loss, *_ = self.diffusion_step(latent_mode, action, next_latent_step_mode.detach(), reward)
            tot_metrics.update(score_metrics)
            self.optim["vae"].zero_grad(set_to_none=True)
            self.optim["diffusion"].zero_grad(set_to_none=True)
            (recon_loss*self.recon_coef + kl_loss*self.kl_coef + diffusion_loss*self.diffusion_coef).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 100)
            self.optim["vae"].step()
            self.optim["diffusion"].step()

            tot_metrics["info/score_grad_norm"] = grad_norm
        self.update_target(tau=1.0) # hard update
        return tot_metrics

    def train_step(self, replay_iter, step):
        self._step += 1
        if self._step % self.update_every != 0:
            return {}
        tot_metrics = {}
        for _ in range(self.extra_repr_step):
            batch = next(replay_iter)
            img_stack, action, reward, discount, next_img_stack, next_img_step = [
                convert_to_tensor(t, self.device) for t in batch
            ]
            img_stack = self.aug(img_stack.float()).detach()
            next_img_stack = self.aug(next_img_stack.float()).detach()
            next_img_step = next_img_step[:, -3:].float().detach()
            ae_metrics, recon_loss, kl_loss, latent, next_latent_step, latent_mode, next_latent_step_mode = \
                self.ae_step(img_stack, next_img_step)
            score_metrics, diffusion_loss, reg_loss, *_ = \
                self.diffusion_step(latent_mode, action, next_latent_step_mode.detach(), reward)
            critic_metrics, critic_loss = \
                self.critic_step(img_stack, action, next_img_stack, reward, discount, step, latent_mode)

            loss = (
                recon_loss * self.recon_coef + \
                kl_loss * self.kl_coef + \
                diffusion_loss * self.diffusion_coef + \
                critic_loss * self.critic_coef
            )

            self.optim["vae"].zero_grad(set_to_none=True)
            self.optim["diffusion"].zero_grad(set_to_none=True)
            self.optim["critic"].zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 100.0)
            self.optim["vae"].step()
            self.optim["diffusion"].step()
            self.optim["critic"].step()

        actor_metrics, actor_loss = self.actor_step(latent_mode.detach(), step)
        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        self.update_target(self.tau)
        tot_metrics.update(ae_metrics)
        tot_metrics.update(score_metrics)
        tot_metrics.update(actor_metrics)
        tot_metrics.update(critic_metrics)
        tot_metrics["info/score_grad_norm"] = grad_norm
        return tot_metrics

    def ae_step(self, img_stack, next_img_step):
        B, S, W, H = img_stack.shape
        img_stack = img_stack.view(B*3, S//3, W, H)
        img_target = torch.concat([img_stack/255. - 0.5, next_img_step/255. - 0.5], dim=0)
        all_img = torch.concat([img_stack, next_img_step], dim=0)
        all_latent, latent_dist, img_pred = self.vae(all_img, sample_posterior=True, forward_decoder=True)
        all_latent_mode = latent_dist.mode()

        # only use some of the data
        img_pred, img_target = img_pred[:B*3], img_target[:B*3]
        recon_loss = F.mse_loss(img_pred, img_target, reduction="sum") / img_pred.shape[0]
        kl_loss = latent_dist.kl()[:B*3].mean()

        latent, next_latent_step = torch.split(all_latent, [B*3, B], dim=0)
        latent_mode, next_latent_step_mode = torch.split(all_latent_mode, [B*3, B], dim=0)
        latent = latent.reshape(B, -1)
        latent_mode = latent_mode.reshape(B, -1)

        return {
            "loss/recon_loss": recon_loss.item(),
            "loss/kl_loss": kl_loss.item(),
            "info/latent_mean": latent.mean().item(),
            "info/latent_std": latent.std().item(),
            "info/latent_l1_norm": latent.abs().mean().item(),
            "info/latent_dist_mean": latent_dist.mode().mean().item(),
            "info/latent_dist_std": latent_dist.std.mean().item()
        }, recon_loss, kl_loss, latent, next_latent_step, latent_mode, next_latent_step_mode

    def diffusion_step(self, latent, action, next_latent_step, reward=None):
        B = latent.shape[0]

        diffusion_loss, stats = self.diffusion.compute_loss(next_latent_step, latent, action)

        # reg loss
        if self.reg_coef != 0:
            raise NotImplementedError
        else:
            reg_loss = torch.tensor(0.0)

        metrics = {
            "loss/reg_loss": reg_loss.item(),
            "loss/diffusion_loss": diffusion_loss.item(),
        }
        metrics.update(stats)
        return metrics, diffusion_loss, reg_loss

    def critic_step(self, img_stack, action, next_img_stack, reward, discount, step, latent):
        B, S, W, H = img_stack.shape
        if self.back_critic_grad:
            feature = self.diffusion.score.forward_psi(latent, action)
        else:
            feature = self.diffusion_target.score.forward_psi(latent, action).detach()

        with torch.no_grad():
            # use target network to encode next latent
            next_img_stack = next_img_stack.view(B*3, S//3, W, H)
            next_latent, _ = self.vae_target(next_img_stack, sample_posterior=False, forward_decoder=False)
            next_latent = next_latent.reshape(B, -1)
            next_latent = self.scaler.forward(next_latent)
            # get target values
            stddev = self.stddev_schedule(step)
            dist = self.actor(next_latent, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_feature = self.diffusion_target.score.forward_psi(next_latent, next_action)
            q_target = self.critic_target(next_feature)
            q_target = reward + discount*q_target.min(0)[0]
        q_pred = self.critic(feature)
        critic_loss = self.critic_loss_fn(q_pred, q_target.unsqueeze(0).repeat(2, 1, 1))

        q_error = (q_target - q_pred).abs()
        return {
            "loss/critic_loss": critic_loss.item(),
            "info/q_pred": q_pred.mean().item(),
            "info/q_target": q_target.mean().item(),
            "info/reward_mean": reward.mean().item(),
            "info/reward_max": reward.max().item(),
            "info/reward_min": reward.min().item(),
            "info/q_error_mean": q_error.mean().item(),
            "info/q_error_max": q_error.max().item(),
            "info/q_error_min":q_error.min().item()
        }, critic_loss

    def actor_step(self, latent, step):
        stddev = self.stddev_schedule(step)
        dist = self.actor(latent, stddev)
        action = dist.sample(clip=self.stddev_clip)
        actor_loss = -self.critic(self.diffusion_target.score.forward_psi(latent, action)).min(0)[0].mean()

        return {
            "loss/actor_loss": actor_loss.item(),
            "info/policy_std": stddev
        }, actor_loss
