import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal 
import os

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from agent.sac.sac_agent import SACAgent
from agent.sac.actor import DiagGaussianActor

from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
	"""
	Critic on top of phi
	"""
	def __init__(
		self,
		feature_dim,
		hidden_dim=1024,
		):

		super().__init__()

		self.feature_dim = feature_dim
		# Q1
		self.l1 = nn.Linear(feature_dim, hidden_dim) # random feature
		self.l2 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(feature_dim, hidden_dim) # random feature
		self.l5 = nn.Linear(hidden_dim, 1)



	def forward(self, z_phi):
		"""
		"""
		assert z_phi.shape[-1] == self.feature_dim

		q1 = F.elu(self.l1(z_phi)) #F.relu(self.l1(x))
		q1 = self.l2(q1) #F.relu(self.l2(q1))

		q2 = F.elu(self.l4(z_phi)) #F.relu(self.l4(x))
		q2 = self.l5(q2) #F.relu(self.l5(q2))

		return q1, q2

class Phi(nn.Module):
	"""
	phi: s, a -> z_phi in R^d
	"""
	def __init__(
		self, 
		state_dim,
		action_dim,
		feature_dim=1024,
		hidden_dim=1024,
		):

		super(Phi, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, feature_dim)

	def forward(self, state, action):
		x = torch.cat([state, action], axis=-1)
		z = F.elu(self.l1(x)) 
		z = F.elu(self.l2(z)) 
		z_phi = self.l3(z)
		return z_phi

class Mu(nn.Module):
	"""
	mu': s' -> z_mu in R^d
	"""
	def __init__(
		self, 
		state_dim,
		feature_dim=1024,
		hidden_dim=1024,
		):

		super(Mu, self).__init__()

		self.l1 = nn.Linear(state_dim , hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, feature_dim)

	def forward(self, state):
		z = F.elu(self.l1(state))
		z = F.elu(self.l2(z)) 
		# bounded mu's output
		z_mu = F.tanh(self.l3(z)) 
		# z_mu = self.l3(z)
		return z_mu

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


class CTRLSACAgent(SACAgent):
	"""
	SAC with VAE learned latent features
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_space, 
			lr=1e-4, # 3e-4 was originally proposed in the paper, but seems to results in fluctuating performance
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=1024,
			feature_tau=0.005,
			feature_dim=2048, # latent feature dim
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

		self.phi = Phi(state_dim=state_dim, 
				 action_dim=action_dim, 
				 feature_dim=feature_dim, 
				 hidden_dim=hidden_dim).to(device)
		if use_feature_target:
			self.phi_target = copy.deepcopy(self.phi)

		self.mu = Mu(state_dim=state_dim,
			   	feature_dim=feature_dim, 
				hidden_dim=hidden_dim).to(device)

		self.theta = Theta(feature_dim=feature_dim).to(device)

		self.feature_optimizer = torch.optim.Adam(
			list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.theta.parameters()),
			weight_decay=0, lr=lr)
		
		# frozen phi for critic/actor update
		self.frozen_phi = Phi(state_dim=state_dim, 
				 action_dim=action_dim, 
				 feature_dim=feature_dim, 
				 hidden_dim=hidden_dim).to(device)
		if use_feature_target:
			self.frozen_phi_target = copy.deepcopy(self.frozen_phi)

		self.actor = DiagGaussianActor(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=256,
			hidden_depth=2,
			log_std_bounds=[-5., 2.], 
		).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
										  weight_decay=0, lr=lr/3, betas=[0.9, 0.999]) 	# lower lr for actor/alpha
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr/3, betas=[0.9, 0.999])
		

		self.critic = Critic(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 
										   weight_decay=0, lr=lr, betas=[0.9, 0.999])

		# print network summary
		batch_size = 256
		summary(self.phi, input_size=[(batch_size, state_dim), (batch_size, action_dim)])
		summary(self.mu, input_size=(batch_size, state_dim))
		summary(self.theta, input_size=(batch_size, feature_dim))
		summary(self.critic, input_size=(batch_size, feature_dim))


	def feature_step(self, batch):
		"""
		Loss implementation 
		"""
		state, action, next_state, reward, _ = unpack_batch(batch)

		z_phi = self.phi(state, action)

		z_mu_next = self.mu(next_state)

		assert z_phi.shape[-1] == self.feature_dim
		assert z_mu_next.shape[-1] == self.feature_dim

		labels = torch.eye(state.shape[0]).to(device)

		# we take NCE gamma = 1 here, the paper uses 0.2
		contrastive = (z_phi[:, None, :] * z_mu_next[None, :, :]).sum(-1) 
		model_loss = nn.CrossEntropyLoss()
		model_loss = model_loss(contrastive, labels)

		r_loss = 0.5 * F.mse_loss(self.theta(z_phi), reward).mean()

		## probability density constraint, not used for now
		# prob_loss = torch.mm(z_phi, z_mu_next.t()).mean(dim=1)
		# prob_loss = (z_phi * z_mu_next).sum(-1).clamp(min=1e-4)
		# prob_loss = prob_loss.log().square().mean()  

		loss = model_loss + r_loss # + prob_loss

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

			if self.use_feature_target:
				z_phi = self.frozen_phi_target(state, action)
				z_phi_next = self.frozen_phi_target(next_state, next_action)
			else:
				z_phi = self.frozen_phi(state, action)
				z_phi_next = self.frozen_phi(next_state, next_action)

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

		z_phi = self.frozen_phi(batch.state, action)

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
		for _ in range(self.extra_feature_steps+1):
			batch = buffer.sample(batch_size)

			feature_info = self.feature_step(batch)

			# Update the feature network if needed
			if self.use_feature_target:
				self.update_feature_target()

		# copy phi to frozen phi
		self.frozen_phi.load_state_dict(self.phi.state_dict().copy())
		if self.use_feature_target:
			self.frozen_phi_target.load_state_dict(self.phi.state_dict().copy())

		# Critic step
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


	
