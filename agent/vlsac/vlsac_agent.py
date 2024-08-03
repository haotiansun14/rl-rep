import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal 
import os

from utils.util import unpack_batch
from networks.policy import GaussianPolicy
from networks.vae import Encoder, Decoder, GaussianFeature
from agent.sac.sac_agent import SACAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
	"""
	Critic with random fourier features
	"""
	def __init__(
		self,
		feature_dim,
		num_noise=20, 
		hidden_dim=256,
		):

		super().__init__()
		self.num_noise = num_noise
		self.noise = torch.randn(
			[self.num_noise, feature_dim], requires_grad=False, device=device)

		# Q1
		self.l1 = nn.Linear(feature_dim, hidden_dim) # random feature
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(feature_dim, hidden_dim) # random feature
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, mean, log_std):
		"""
		"""
		std = log_std.exp()
		batch_size, d = mean.shape 
	
		x = mean[:, None, :] + std[:, None, :] * self.noise
		x = x.reshape(-1, d)

		q1 = F.elu(self.l1(x)) #F.relu(self.l1(x))
		q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
		q1 = F.elu(self.l2(q1)) #F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.elu(self.l4(x)) #F.relu(self.l4(x))
		q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
		q2 = F.elu(self.l5(q2)) #F.relu(self.l5(q2))
		q2 = self.l3(q2)

		return q1, q2



class VLSACAgent(SACAgent):
	"""
	SAC with VAE learned latent features
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_space, 
			lr=1e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=256,
			feature_tau=0.001,
			feature_dim=256, # latent feature dim
			use_feature_target=True, 
			extra_feature_steps=1,
			):

		super().__init__(
			state_dim=state_dim,
			action_dim=action_dim,
			action_space=action_space,
			lr=lr,
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

		self.encoder = Encoder(state_dim=state_dim, 
				action_dim=action_dim, feature_dim=feature_dim).to(device)
		self.decoder = Decoder(state_dim=state_dim,
				feature_dim=feature_dim).to(device)
		self.f = GaussianFeature(state_dim=state_dim, 
				action_dim=action_dim, feature_dim=feature_dim).to(device)
		
		if use_feature_target:
			self.f_target = copy.deepcopy(self.f)
		self.feature_optimizer = torch.optim.Adam(
			list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.f.parameters()),
			lr=lr)

		self.critic = Critic(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=lr, betas=[0.9, 0.999])


	def feature_step(self, batch):
		"""
		Feature learning step

		KL between two gaussian p1 and p2:

		log sigma_2 - log sigma_1 + sigma_1^2 (mu_1 - mu_2)^2 / 2 sigma_2^2 - 0.5
		"""
		# ML loss
		z = self.encoder.sample(
			batch.state, batch.action, batch.next_state)
		x, r = self.decoder(z)
		s_loss = 0.5 * F.mse_loss(x, batch.next_state)
		r_loss = 0.5 * F.mse_loss(r, batch.reward)
		ml_loss = r_loss + s_loss

		# KL loss
		mean1, log_std1 = self.encoder(
			batch.state, batch.action, batch.next_state)
		mean2, log_std2 = self.f(batch.state, batch.action)
		var1 = (2 * log_std1).exp()
		var2 = (2 * log_std2).exp()
		kl_loss = log_std2 - log_std1 + 0.5 * (var1 + (mean1-mean2)**2) / var2 - 0.5
		
		loss = (ml_loss + kl_loss).mean()

		self.feature_optimizer.zero_grad()
		loss.backward()
		self.feature_optimizer.step()

		return {
			'vae_loss': loss.item(),
			'ml_loss': ml_loss.mean().item(),
			'kl_loss': kl_loss.mean().item(),
			's_loss': s_loss.mean().item(),
			'r_loss': r_loss.mean().item()
		}


	def update_actor_and_alpha(self, batch):
		"""
		Actor update step
		"""
		dist = self.actor(batch.state)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		if self.use_feature_target:
			mean, log_std = self.f_target(batch.state, action)
		else:
			mean, log_std = self.f(batch.state, action)
		q1, q2 = self.critic(mean, log_std)
		q = torch.min(q1, q2)

		actor_loss = ((self.alpha) * log_prob - q).mean()

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


	def critic_step(self, batch):
		"""
		Critic update step
		"""			
		state, action, next_state, reward, done = unpack_batch(batch)

		with torch.no_grad():
			dist = self.actor(next_state)
			next_action = dist.rsample()
			next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)

			if self.use_feature_target:
				mean, log_std = self.f_target(state, action)
				next_mean, next_log_std = self.f_target(next_state, next_action)
			else:
				mean, log_std = self.f(state, action)
				next_mean, next_log_std = self.f(next_state, next_action)

			next_q1, next_q2 = self.critic_target(next_mean, next_log_std)
			next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
			target_q = reward + (1. - done) * self.discount * next_q 
			
		q1, q2 = self.critic(mean, log_std)
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


	def update_feature_target(self):
		for param, target_param in zip(self.f.parameters(), self.f_target.parameters()):
			target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)
	

	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		# Feature step
		for _ in range(self.extra_feature_steps+1):
			batch = buffer.sample(batch_size)
			feature_info = self.feature_step(batch)

			# Update the feature network if needed
			if self.use_feature_target:
				self.update_feature_target()

		# Acritic step
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


	
