import torch
import torch.nn.functional as F
import numpy as np

from network_arch.drqv2 import RandomShiftsAug, Encoder, Decoder, Critic, Actor
from helper_functions.util import make_target, convert_to_tensor
from UtilsRL.misc.decorator import profile
import torchvision
from helper_functions.util import sync_target
from helper_functions.util import setup_schedule

class DrQv2():
    def __init__(
        self, 
        obs_space, 
        action_space, 
        args
    ) -> None:
        super().__init__()
        self.args = args
        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape[0]
        self.action_range = [
            action_space.minimum, 
            action_space.maximum
        ]
        self.tau = args.tau
        self.update_every = args.update_every
        self.device = args.device
        self.critic_loss = args.critic_loss
        if self.critic_loss == "mse":
            self.critic_loss_fn = F.mse_loss
        elif self.critic_loss == "huber":
            self.critic_loss_fn = F.huber_loss
            
        # setup stddev schedule
        self.stddev_schedule = setup_schedule(args.stddev_schedule)
        self.stddev_clip = args.stddev_clip
        # networks related
        self.bn_dim = args.bn_dim
        self.encoder = Encoder(self.obs_dim).to(self.device)
        self.actor = Actor(
            repr_dim=self.encoder.repr_dim, 
            action_dim=self.action_dim, 
            bn_dim=self.bn_dim, 
            hidden_dim=args.actor_hidden_dim, 
        ).to(self.device)
        self.critic = Critic(
            repr_dim=self.encoder.repr_dim, 
            action_dim=self.action_dim, 
            bn_dim=self.bn_dim, 
            hidden_dim=args.critic_hidden_dim
        ).to(self.device)
        self.critic_target = make_target(self.critic)
        self.aug = RandomShiftsAug(pad=4)

        # optimizers related
        self.optim = {
            "encoder": torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr), 
            "actor": torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr), 
            "critic": torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr), 
        }
        self._step = 1
        self.train()
        
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
    
    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        obs = obs.view(1, -1)
        stddev = self.stddev_schedule(step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=None) if not deterministic else dist.mean
        return action.squeeze().cpu().numpy()
    
    def update_target(self):
        sync_target(self.critic, self.critic_target, self.tau)
    
    @torch.no_grad()
    def evaluate(self, replay_iter):
        return {}, None
        
    def pretrain_step(self, replay_iter, step):
        return {}
                
    def train_step(self, replay_iter, step):
        self._step += 1
        if self._step % self.update_every != 0:
            return {}
        tot_metrics = {}
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = self.aug(img_stack.float())
        next_img_stack = self.aug(next_img_stack.float())
        
        latent_stack, critic_metrics = self.critic_step(img_stack, action, next_img_stack, reward, discount, step)
        actor_metrics = self.actor_step(latent_stack.detach(), step)
        self.update_target()
        tot_metrics.update(actor_metrics)
        tot_metrics.update(critic_metrics)
        return tot_metrics
    
    def critic_step(self, img_stack, action, next_img_stack, reward, discount, step):
        latent_stack = self.encoder(img_stack)
        next_latent_stack = self.encoder(next_img_stack).detach()
        with torch.no_grad():
            stddev = self.stddev_schedule(step)
            dist = self.actor(next_latent_stack, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            q_target = self.critic_target(next_latent_stack, next_action)
            q_target = reward + discount*q_target.min(0)[0]
        q_pred = self.critic(latent_stack, action)
        critic_loss = self.critic_loss_fn(q_pred, q_target.unsqueeze(0).repeat(2, 1, 1))
        self.optim["critic"].zero_grad()
        self.optim["encoder"].zero_grad()
        critic_loss.backward()
        self.optim["critic"].step()
        self.optim["encoder"].step()
        
        return latent_stack, {
            "loss/critic_loss": critic_loss.item(), 
            "info/q_pred": q_pred.mean().item(), 
            "info/q_target": q_target.mean().item(), 
            "info/reward": reward.mean().item()
        }
    
    def actor_step(self, latent_stack, step):
        stddev = self.stddev_schedule(step)
        dist = self.actor(latent_stack, stddev)
        action = dist.sample(clip=self.stddev_clip)
        actor_loss = -self.critic(latent_stack, action).min(0)[0].mean()
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()
        
        return {
            "loss/actor_loss": actor_loss.item(), 
            "info/policy_std": stddev
        }

    