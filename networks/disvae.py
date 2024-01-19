import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
  """
  Discrete encoder

  s,a,s' -> z
  """
  def __init__(
    self,
    state_dim,
    action_dim,
    category_size=16,
    class_size=16,
    hidden_dim=256,
  ):
  
    super(Encoder, self).__init__()

    feature_dim = category_size * class_size 
    self.category_size = category_size
    self.class_size = class_size

    self.l1 = nn.Linear(state_dim + action_dim + state_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(feature_dim) # get logits


  def forward(self, state, action, next_state): 
    """
    get logits
    """
    x = torch.cat([state, action, next_state], axis=-1)
    z = F.relu(self.l1(x)) 
    z = F.relu(self.l2(z)) 
    logit = F.self.l3(z) 
    return logit 


  def get_dist(self, state, action, next_state):
    """
    Get the distribution over category * class
    """
    batch_size = state.shape[0] 
    logit = self.forward(state, action, next_state)
    logit = torch.reshape(logit, shape=(batch_size, self.category_size, self.class_size))
    dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
    return dist 



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
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.state_linear = nn.Linear(hidden_dim, state_dim)
    self.reward_linear = nn.Linear(hidden_dim, 1)


  def forward(self, feature):
    """
    Decode an input feature to observation
    """
    x = F.relu(self.l1(feature))
    x = F.relu(self.l2(x))
    s = self.state_linear(x)
    r = self.reward_linear(x)
    return s, r


class DiscreteFeature(nn.Module):
  """
  Feature extraction 

  s,a -> z
  """
  def __init__(
    self,
    state_dim,
    action_dim,
    category_size=16,
    class_size=16,
    hidden_dim=256,
  ):
    super(DiscreteFeature, self).__init__()

    feature_dim = category_size * class_size 
    self.category_size = category_size
    self.class_size = class_size

    self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, feature_dim)

  
  def forward(self, state, action):
    """
    """
    x = torch.cat([state, action], axis=-1)
    z = F.relu(self.l1(x))
    z = F.relu(self.l2(z))
    logit = self.l3(z)
    return logit


  def get_dist(self, state, action):
    """
    Get the distribution over category * class
    """
    batch_size = state.shape[0] 
    logit = self.forward(state, action)
    logit = torch.reshape(logit, shape=(batch_size, self.category_size, self.class_size))
    dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
    return dist 


  def get_feature(self, state, action):
    batch_size = state.shape[0]
    logit = self.forward(state, action)
    logit = torch.reshape(logit, shape=(batch_size, self.category_size, self.class_size))
    dist = td.OneHotCategoricalStraightThrough(logits=logit)
    features = dist.probs.reshape(batch_size, -1)
    return features 
