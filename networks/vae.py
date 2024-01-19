import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Encoder(nn.Module):
  """
  Gaussian encoder

  s,a,s' -> z
  """
  def __init__(
    self, 
    state_dim,
    action_dim,
    feature_dim=256,
    hidden_dim=256,
    ):

    super(Encoder, self).__init__()

    input_dim = state_dim + action_dim + state_dim
    self.l1 = nn.Linear(input_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)

    self.mean_linear = nn.Linear(hidden_dim, feature_dim)
    self.log_std_linear = nn.Linear(hidden_dim, feature_dim)


  def forward(self, state, action, next_state):
    """
    """
    x = torch.cat([state, action, next_state], axis=-1)

    z = F.relu(self.l1(x)) 
    z = F.relu(self.l2(z)) 
    mean = self.mean_linear(z)
    log_std = self.log_std_linear(z)
    log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

    return mean, log_std

  def sample(self, state, action, next_state):
    """
    """
    mean, log_std = self.forward(state, action, next_state)
    std = log_std.exp()
    normal = Normal(mean, std)
    z = normal.rsample() # reparameterization
    return z 


class Decoder(nn.Module):
  """
  Deterministic decoder (Gaussian with identify covariance)

  z -> s
  """
  def __init__(
    self, 
    state_dim,
    feature_dim=256,
    hidden_dim=256,):

    super(Decoder, self).__init__()

    self.l1 = nn.Linear(feature_dim, hidden_dim)
    self.state_linear = nn.Linear(hidden_dim, state_dim)
    self.reward_linear = nn.Linear(hidden_dim, 1)


  def forward(self, feature):
    """
    Decode an input feature to observation
    """
    x = F.relu(self.l1(feature)) #F.relu(self.l1(feature))
    s = self.state_linear(x)
    r = self.reward_linear(x)
    return s, r


class GaussianFeature(nn.Module):
  """
  Gaussian feature extraction with parameterized mean and std

  s,a -> z 
  """
  def __init__(
    self,
    state_dim, 
    action_dim,
    feature_dim=256, 
    hidden_dim=256, 
  ):

    super(GaussianFeature, self).__init__()

    self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.mean_linear = nn.Linear(hidden_dim, feature_dim)
    self.log_std_linear = nn.Linear(hidden_dim, feature_dim)


  def forward(self, state, action):
    x = torch.cat([state, action], axis=-1)

    z = F.relu(self.l1(x)) 
    z = F.relu(self.l2(z)) 
    mean = self.mean_linear(z)
    log_std = self.log_std_linear(z)
    log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

    return mean, log_std

