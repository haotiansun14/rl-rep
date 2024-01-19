import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import util

from agent.sac.critic import DoubleQCritic
from agent.sac.actor import DiagGaussianActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent(object):
	"""
	DDPG Agent
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_space, 
			lr=3e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=1024,
			):

		self.steps = 0

		self.device = device 
		self.action_range = [
			float(action_space.low.min()),
			float(action_space.high.max())
		]
		self.discount = discount 
		self.tau = tau 
		self.target_update_period = target_update_period
		self.learnable_temperature = auto_entropy_tuning

		# functions
		self.critic = DoubleQCritic(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
		).to(self.device)
		self.critic_target = DoubleQCritic(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
		).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.actor = DiagGaussianActor(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
			log_std_bounds=[-5., 2.], 
		).to(self.device)
		self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -action_dim
		
		 # optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
																						lr=lr,
																						betas=[0.9, 0.999])

		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
																							lr=lr,
																							betas=[0.9, 0.999])

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
																								lr=lr,
																								betas=[0.9, 0.999])


	@property
	def alpha(self):
		return self.log_alpha.exp()


	def select_action(self, state, explore=False):
		state = torch.FloatTensor(state).to(self.device)
		state = state.unsqueeze(0)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(*self.action_range)
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])


	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def critic_step(self, batch):
		"""
		Critic update step
		"""
		obs, action, next_obs, reward, done = util.unpack_batch(batch)
		not_done = 1. - done

		dist = self.actor(next_obs)
		next_action = dist.rsample()
		log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
		target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
		target_V = torch.min(target_Q1,
													target_Q2) - self.alpha.detach() * log_prob
		target_Q = reward + (not_done * self.discount * target_V)
		target_Q = target_Q.detach()

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
				current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return {
			'q_loss': critic_loss.item(), 
			'q1': current_Q1.mean().item(),
			'q2': current_Q1.mean().item()
			}


	def update_actor_and_alpha(self, batch):
		obs = batch.state 

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_Q1, actor_Q2 = self.critic(obs, action)

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


	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)
		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			**critic_info, 
			**actor_info,
		}
