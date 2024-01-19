import torch
from torch import nn
from torch.nn import functional as F


class ValueCritic(nn.Module):
	"""
	MLP V-network
	"""
	def __init__(
		self, 
		state_dim, 
		hidden_dim=256,
	):

		super(ValueCritic, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)


	def forward(self, state):
		v = F.relu(self.l1(state))
		v = F.relu(self.l2(v))
		v = self.l3(v)
		return v 

		

class Critic(nn.Module):
	"""
	MLP Double Q-network
	"""
	def __init__(
		self,
		state_dim,
		action_dim,
		hidden_dim=256,
	):

		super(Critic, self).__init__()

		# Q1
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)
	

	def forward(self, state, action):
		x = torch.cat([state, action], axis=-1)

		q1 = F.relu(self.l1(x)) 
		q1 = F.relu(self.l2(q1)) 
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(x)) 
		q2 = F.relu(self.l5(q2)) 
		q2 = self.l6(q2)

		return q1, q2


class LinearCritic(nn.Module):
	"""
	Critic with linear feature as input
	"""
	def __init__(
		self,
		feature_dim,
		hidden_dim=256):

		super(LinearCritic, self).__init__()

		# Q1
		self.l1 = nn.Linear(feature_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(feature_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, x):
		q1 = F.relu(self.l1(x))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(x))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		return q1, q2



class RFFLinearCritic(nn.Module):
	"""
	Critic with random fourier features
	"""
	def __init__(
		self,
		feature_dim,
		num_rff=1024,
		hidden_dim=256,
		train_rff_weights=True, # if train the RFF weights
		):

		super(RFFLinearCritic, self).__init__()

		# Q1
		self.l1 = nn.Linear(feature_dim, num_rff) # random feature
		self.l2 = nn.Linear(num_rff, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(feature_dim, num_rff) # random feature
		self.l5 = nn.Linear(num_rff, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)

		# set random weights for random feature
		w = torch.randn_like(self.l1.weight)
		b = torch.rand_like(self.l1.bias) * 2*3.1415926

		with torch.no_grad():
			self.l1.weight.copy_(w)
			self.l1.bias.copy_(b)
			self.l4.weight.copy_(w)
			self.l4.bias.copy_(b)


	def forward(self, x):
		q1 = F.relu(self.l1(x))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(x))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		return q1, q2



