import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal 
import numpy as np 

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianPolicy(nn.Module):
	"""
	Gaussian policy
	"""
	def __init__(
		self, 
		state_dim,
		action_dim,
		action_space,
		hidden_dim=256,
	):
		super(GaussianPolicy, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)

		self.mean_linear = nn.Linear(hidden_dim, action_dim)
		self.log_std_linear = nn.Linear(hidden_dim, action_dim)

		self.action_scale = torch.FloatTensor(
			(action_space.high - action_space.low) / 2.).to(device)
		self.action_bias = torch.FloatTensor(
			(action_space.high + action_space.low) / 2.).to(device)


	def forward(self, state):
		"""
		"""
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mean = self.mean_linear(a)
		log_std = self.log_std_linear(a)
		log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
		return mean, log_std
	

	# def sample(self, state):
	# 	"""
	# 	Sample an action from the policy
	# 	"""
	# 	mean, log_std = self.forward(state)
	# 	std = log_std.exp()
	# 	normal = Normal(mean, std)

	# 	x_t = normal.rsample().to(device) # reparameterization
	# 	y_t = torch.tanh(x_t)
	# 	action = y_t * self.action_scale + self.action_bias
	# 	log_prob = normal.log_prob(x_t)

	# 	# correction: enforcing action bound.
	# 	correction = - 2. * (
	# 		torch.from_numpy(np.log([2.])).to(device)
	# 		- x_t 
	# 		- F.softplus(-2. * x_t)
	# 	)
	# 	log_prob += correction 
	# 	log_prob = log_prob.sum(1, keepdim=True)
		
	# 	# get mean
	# 	mean = torch.tanh(mean) * self.action_scale + self.action_bias

	# 	return action, log_prob, mean

	def sample(self, state):
		"""
		Sample an action from the policy
		"""
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = Normal(mean, std)

		x_t = normal.rsample() # reparameterization
		y_t = torch.tanh(x_t)
		action = y_t * self.action_scale + self.action_bias
		log_prob = normal.log_prob(x_t)

		# enforcing action bound. See Appendix C of the SAC paper
		log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
		log_prob = log_prob.sum(1, keepdim=True)
		mean = torch.tanh(mean) * self.action_scale + self.action_bias

		return action, log_prob, mean
