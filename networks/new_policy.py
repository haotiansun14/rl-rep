import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal 
from torch import distributions as pyd
from torch.distributions import Distribution as TorchDistribution


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Distribution(TorchDistribution):
	def sample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p

	def rsample_and_logprob(self):
		s = self.rsample()
		log_p = self.log_prob(s)
		return s, log_p

	def mle_estimate(self):
		return self.mean

	def get_diagnostics(self):
		return {}


class TanhNormal(Distribution):
	"""
	Represent distribution of X
		X ~ tanh(Z)
		Z ~ N(mean, std)
	"""
	

class TanhTransform(pyd.transforms.Transform):
	domain = pyd.constraints.real
	codomain = pyd.constraints.interval(-1.0, 1.0)
	bijective = True
	sign = +1

	def __init__(self, cache_size=1):
		super().__init__(cache_size=cache_size)

	@staticmethod
	def atanh(x):
		return 0.5 * (x.log1p() - (-x).log1p())

	def __eq__(self, other):
		return isinstance(other, TanhTransform)

	def _call(self, x):
		return x.tanh()

	def _inverse(self, y):
		# We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
		# one should use `cache_size=1` instead
		return self.atanh(y)

	def log_abs_det_jacobian(self, x, y):
		# We use a formula that is more numerically stable, see details in the following link
		# https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
		return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
	def __init__(self, loc, scale):
		self.loc = loc
		self.scale = scale

		self.base_dist = pyd.Normal(loc, scale)
		transforms = [TanhTransform()]
		super().__init__(self.base_dist, transforms)

	@property
	def mean(self):
		mu = self.loc
		for tr in self.transforms:
				mu = tr(mu)
		return mu


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
		log_prob_ = log_prob

		# correction: enforcing action bound.
		correction = - 2. * (
			torch.from_numpy(np.log([2.]))
			- x_t 
			- F.softplus(-2. * x_t)
		).sum(dim=1)
		
		print(x_t.shape, log_prob.shape, correction.shape)
		log_prob += correction
		print(log_prob.shape, '!!!!!')

		# enforcing action bound. See Appendix C of the SAC paper
		log_prob_ -= torch.log((1 - y_t.pow(2)) + epsilon)
		log_prob_ = log_prob_.sum(1, keepdim=True)

		print(log_prob_.shape, '????')

		mean = torch.tanh(mean) * self.action_scale + self.action_bias

		return action, log_prob, mean


	# def sample(self, state):
	# 	"""
	# 	Sample an action from the policy
	# 	"""
	# 	mean, log_std = self.forward(state)
	# 	std = log_std.exp()
	# 	normal = Normal(mean, std)

	# 	x_t = normal.rsample() # reparameterization
	# 	y_t = torch.tanh(x_t)
	# 	action = y_t * self.action_scale + self.action_bias
	# 	log_prob = normal.log_prob(x_t)

	# 	# enforcing action bound. See Appendix C of the SAC paper
	# 	log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
	# 	log_prob = log_prob.sum(1, keepdim=True)
	# 	mean = torch.tanh(mean) * self.action_scale + self.action_bias

	# 	return action, log_prob, mean
