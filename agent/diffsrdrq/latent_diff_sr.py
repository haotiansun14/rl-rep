import torch
import torch.nn.functional as F
import numpy as np

from helper_functions.util import make_target, convert_to_tensor, generate_alphas_and_alphabars
from UtilsRL.misc.decorator import profile
import torchvision
from helper_functions.util import sync_target, setup_schedule
from network_arch.vae_1d import VAE, Scaler
from network_arch.latent_diff_sr import RandomShiftsAug, Actor, RFFCritic
from network_arch.score_idql import IDQLFactoredScoreNet

class LatentDiffSRDrQv2():
    def __init__(self, obs_space, action_space, args):
        super().__init__()
        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape[0]
        self.action_range = [
            action_space.minimum, 
            action_space.maximum
        ]
        # params
        self.device = args.device
        self.use_repr_target = args.use_repr_target
        self.kl_coef = args.kl_coef
        self.reg_coef = args.reg_coef
        self.ae_coef = args.ae_coef
        self.repr_coef = args.repr_coef
        self.tau = args.tau
        self.grad_norm = args.grad_norm
        
        self.extra_repr_step = args.extra_repr_step
        self.update_every = args.update_every
        self.stddev_schedule = setup_schedule(args.stddev_schedule)
        self.stddev_clip = args.stddev_clip
        self.pretrain_steps = args.pretrain_steps
        self.ae_pretrain_steps = args.ae_pretrain_steps
        self.back_critic_grad = args.back_critic_grad
        self.critic_loss = args.critic_loss
        if self.critic_loss == "mse":
            self.critic_loss_fn = F.mse_loss
        elif self.critic_loss == "huber":
            self.critic_loss_fn = F.huber_loss
            
        # noise related
        self.num_noises = args.num_noises
        self.betas, self.alphas, self.alphabars = generate_alphas_and_alphabars(
            args.noise_param1, 
            args.noise_param2, 
            args.num_noises, 
            args.noise_schedule
        )
        self.betas = self.betas.to(self.device)
        self.alphabars = self.alphabars.to(self.device)
        self.alphabars_prev = F.pad(self.alphabars[:-1], (1, 0), value=1.)
        self.alphas = self.alphas.to(self.device)
        self.betas = self.betas[..., None]
        self.alphabars = self.alphabars[..., None]
        self.alphabars_prev = self.alphabars_prev[..., None]
        self.alphas = self.alphas[..., None]
        
        # networks related
        self.latent_dim = args.latent_dim
        self.feature_dim = args.feature_dim
        self.bn_dim = args.bn_dim
        # VAE structure
        self.vae = VAE(
            obs_shape=self.obs_dim, 
            latent_dim=self.latent_dim, 
            ae_num_filters=args.ae_num_filters,
            ae_num_layers=args.ae_num_layers
        ).to(self.device)
        self.scaler = Scaler(activate=args.do_scale)
        # score networks
        self.score = IDQLFactoredScoreNet(
            latent_dim=self.latent_dim, 
            action_dim=self.action_dim, 
            bn_dim=self.bn_dim, 
            feature_dim=self.feature_dim, 
            psi_hidden_dim=args.psi_hidden_dim, 
            psi_hidden_depth=args.psi_hidden_depth, 
            zeta_hidden_dim=args.zeta_hidden_dim, 
            zeta_hidden_depth=args.zeta_hidden_depth
        ).to(self.device)
        # rl networks
        self.actor = Actor(
            repr_dim=self.latent_dim*3, 
            action_dim=self.action_dim, 
            bn_dim=self.bn_dim, 
            hidden_dim=args.actor_hidden_dim
        ).to(self.device)
        self.critic = RFFCritic(
            input_dim=self.feature_dim, 
            hidden_dim=args.critic_hidden_dim
        ).to(self.device)
        # target networks
        self.critic_target = make_target(self.critic)
        if self.use_repr_target:
            self.vae_target = make_target(self.vae)
            self.score_target = make_target(self.score)
        else:
            self.vae_target = self.vae
            self.score_target = self.score
        self.aug = RandomShiftsAug(pad=4)
        
        # optimizers
        self.optim = {
            "vae": torch.optim.Adam(self.vae.parameters(), lr=args.ae_lr), 
            "score": torch.optim.AdamW(self.score.parameters(), lr=args.score_lr), 
            "actor": torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr), 
            "critic": torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr), 
        }
        self._step = 1
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.vae.train(training)
        self.score.train(training)
        
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
            sync_target(self.score, self.score_target, tau)
    
    @torch.no_grad()
    def evaluate(self, replay_iter):
        self.train(False)
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = img_stack.float()
        next_img_step = next_img_step[:, -3:, ...].float()

        B, S, W, H = img_stack.shape
        img_stack = img_stack.view(B*3, S//3, W, H)
        latent_stack, _ = self.vae_target(img_stack, sample_posterior=False, forward_decoder=False)
        latent_stack = latent_stack.reshape(B, -1)
        next_latent, _ = self.vae_target(next_img_step, sample_posterior=False, forward_decoder=False)
        raw_next_img = next_img_step[:9, :] / 255. - 0.5
        recon_next_img = self.vae_target.decode(next_latent)[:9, :]

        # DDPM sampling
        latent_stack = self.scaler(latent_stack)
        next_latent = self.scaler(next_latent)
        psi = self.score_target.forward_psi(latent_stack, action)
        xt = torch.randn_like(next_latent)
        stepsize = self.num_noises // 5
        checkpoints = list(range(0, self.num_noises, stepsize)) + [self.num_noises]
        checkpoints = list(reversed(checkpoints))
        checkpoint_next_latents = [
            xt, 
        ]
        checkpoint_gen_next_imgs = [
            self.vae_target.decode(xt)[:9, :]
        ]
        
        def reverse(xt, t):
            z = torch.randn_like(next_latent)
            timestep = t * torch.ones([xt.shape[0], ], dtype=torch.int64).to(self.device)
            score = self.score_target.forward_score(xt, timestep, psi=psi)
            sigma_t = 0
            if t > 0:
                sigma_t_square = self.betas[timestep]*(1-self.alphabars_prev[timestep]) / (1-self.alphabars[timestep])
                sigma_t_square = sigma_t_square.clip(1e-20)
                sigma_t = sigma_t_square.sqrt()
            return 1. / self.alphas[timestep].sqrt() * (xt + self.betas[timestep] * score) + sigma_t * z
        
        for t in range(self.num_noises-1, -1, -1):
            xt = reverse(xt, t)
            xt = torch.clip(xt, min=-7., max=7.)
            if t in checkpoints:
                checkpoint_next_latents.append(xt)
                checkpoint_gen_next_imgs.append(self.vae_target.decode(self.scaler(xt, reverse=True))[:9, :])
        
        checkpoint_next_latents = torch.stack(checkpoint_next_latents, dim=0)
        img_to_show = torch.stack([raw_next_img, recon_next_img]+checkpoint_gen_next_imgs, dim=0)
        latent_l1_diff = (checkpoint_next_latents - next_latent.repeat(6, 1, 1, 1, 1)).abs().mean([1,2,3,4])
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
    
    def pretrain_step(self, replay_iter, step):
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = self.aug(img_stack.float()).detach()
        next_img_step = next_img_step[:, -3:].float().detach()
        tot_metrics = {}
        if step < self.ae_pretrain_steps:
            metrics, ae_loss, *_ = self.ae_step(img_stack, next_img_step)
            tot_metrics.update(metrics)
            self.optim["vae"].zero_grad(set_to_none=True)
            (ae_loss*self.ae_coef).backward()
            self.optim["vae"].step()
        else:
            metrics, ae_loss, latent, next_latent_step, latent_mode = self.ae_step(img_stack, next_img_step)
            tot_metrics.update(metrics)
            metrics, score_loss, reg_loss, *_ = self.score_step(latent, action, next_latent_step, reward)
            tot_metrics.update(metrics)
            self.optim["vae"].zero_grad(set_to_none=True)
            self.optim["score"].zero_grad(set_to_none=True)
            (ae_loss*self.ae_coef + reg_loss*self.reg_coef + score_loss).backward()
            self.optim["vae"].step()
            self.optim["score"].step()            
        self.update_target(tau=1.0) # hard update
        return tot_metrics

    def ae_step(self, img_stack, next_img_step):
        B, S, W, H = img_stack.shape
        img_stack = img_stack.view(B*3, S//3, W, H)
        img_target = torch.concat([img_stack/255. - 0.5, next_img_step/255. - 0.5], dim=0)
        all_img = torch.concat([img_stack, next_img_step], dim=0)
        all_latent, latent_dist, img_pred = self.vae(all_img, sample_posterior=True, forward_decoder=True)
        all_latent_mode = latent_dist.mode()

        recon_loss = F.mse_loss(img_pred, img_target, reduction="sum") / img_pred.shape[0]
        kl_loss = latent_dist.kl().mean()
        ae_loss = recon_loss + self.kl_coef*kl_loss

        latent, next_latent_step = torch.split(all_latent, [B*3, B], dim=0)
        latent_mode, next_latent_mode = torch.split(all_latent_mode, [B*3, B], dim=0)
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
        }, ae_loss, latent, next_latent_step, latent_mode
        
    def get_reg_loss(self, psi):
        B = psi.shape[0]
        
        def reg_loss_for_batch(feature):
            inprods = torch.mm(feature, feature.transpose(1, 0))
            norms = inprods[torch.arange(B), torch.arange(B)]
            part1 = inprods.pow(2).sum() - norms.pow(2).sum()
            part1 /= B
            part2 = -2 * norms.mean()
            return part1 + part2
        feature1, feature2 = self.critic.forward_feature(psi)
        reg_loss = reg_loss_for_batch(feature1) + reg_loss_for_batch(feature2)
        return reg_loss
        
    def score_step(self, latent, action, next_latent_step, reward=None):
        B = latent.shape[0]
        
        # score matching
        noise_idx = torch.randint(0, self.num_noises, (B, )).to(self.device)
        alphabars = self.alphabars[noise_idx]
        noise = torch.randn_like(next_latent_step).to(self.device)
        next_latent_perturbed = alphabars.sqrt()*next_latent_step + (1-alphabars).sqrt()*noise
        psi = self.score.forward_psi(latent, action)
        score = self.score.forward_score(next_latent_perturbed, noise_idx, psi=psi)
        score_loss = (score*(1-alphabars).sqrt() + noise).pow(2).sum(1).mean()
        
        # reg loss
        if self.reg_coef != 0:
            reg_loss = self.get_reg_loss(psi)
        else:
            reg_loss = torch.tensor(0.0)
            
        # reward reconstruction
        # if self.reward_coef != 0:
        #     reward_pred = self.reward_decoder(phi_out)
        #     reward_loss = (reward_pred-reward).pow(2).mean()
        # else:
        #     reward_loss = torch.tensor(0.0)

        return {
            "loss/reg_loss": reg_loss.item(), 
            "loss/score_loss": score_loss.item(), 
            "info/psi_l1_norm": psi.abs().mean().item(),        
        }, score_loss, reg_loss
                
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
            ae_metrics, ae_loss, latent, next_latent_step, latent_mode = self.ae_step(img_stack, next_img_step)
            score_metrics, score_loss, reg_loss = self.score_step(latent, action, next_latent_step, reward)
            critic_metrics, critic_loss = self.critic_step(img_stack, action, next_img_stack, reward, discount, step, latent_mode)

            loss = ((ae_loss*self.ae_coef) + (reg_loss*self.reg_coef) + score_loss) * self.repr_coef + critic_loss
            
            self.optim["vae"].zero_grad(set_to_none=True)
            self.optim["score"].zero_grad(set_to_none=True)
            self.optim["critic"].zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_norm:
                vae_grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_norm)
                score_grad_norm = torch.nn.utils.clip_grad_norm_(self.score.parameters(), self.grad_norm)
            else:
                vae_grad_norm = score_grad_norm = torch.tensor(0.0)
            self.optim["vae"].step()
            self.optim["score"].step()
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
        tot_metrics.update({
            "info/vae_grad_norm": vae_grad_norm.item(),
            "info/score_grad_norm": score_grad_norm.item(),
        })
        return tot_metrics
    
    def critic_step(self, img_stack, action, next_img_stack, reward, discount, step, latent):
        B, S, W, H = img_stack.shape
        if self.back_critic_grad:
            feature = self.score.forward_psi(latent, action)
        else:
            feature = self.score_target.forward_psi(latent, action).detach()
                
        with torch.no_grad():
            next_latent = self.encode_imgs(next_img_stack, use_target=True)
            stddev = self.stddev_schedule(step)
            dist = self.actor(next_latent, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_feature = self.score_target.forward_psi(next_latent, next_action)
            q_target = self.critic_target(next_feature)
            q_target = reward + discount*q_target.min(0)[0]
            
        q_pred = self.critic(feature)
        critic_loss = self.critic_loss_fn(q_pred, q_target.unsqueeze(0).repeat(2, 1, 1))

        return {
            "loss/critic_loss": critic_loss.item(), 
            "info/q_pred": q_pred.mean().item(), 
            "info/q_target": q_target.mean().item(), 
            "info/reward": reward.mean().item()
        }, critic_loss
    
    def actor_step(self, latent, step):
        stddev = self.stddev_schedule(step)
        dist = self.actor(latent, stddev)
        action = dist.sample(clip=self.stddev_clip)
        actor_loss = -self.critic(self.score_target.forward_psi(latent, action)).min(0)[0].mean()
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "info/policy_std": stddev
        }, actor_loss

    def encode_imgs(self, a, b=None, use_target=False):
        B, S, W, H = a.shape
        if b is None:
            all_img = a
            all_B = B
        else:
            all_img = torch.concat([a, b], dim=0)
            all_B = B*2
        all_img = all_img.view(all_B*3, S//3, W, H)
        if use_target:
            all_latent, _ = self.vae_target(all_img, forward_decoder=False, sample_posterior=False)
        else:
            all_latent, _ = self.vae(all_img, forward_decoder=False, sample_posterior=False)
        all_latent = all_latent.reshape(all_B, -1)
        all_latent = self.scaler.forward(all_latent)
        if b is None:
            return all_latent
        else:
            return torch.chunk(all_latent, 2, dim=0)
        
