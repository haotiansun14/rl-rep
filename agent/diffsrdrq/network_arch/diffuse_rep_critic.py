from torch import nn
import torch

from helper_functions import util



class SinElu(nn.Module):
    def forward(self, input):
        return 42.2 * torch.sin(0.03 * input - 1.09) + 39.76


class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class RFFSweepCritic(nn.Module):

    def __init__(self, critic_feed_feature_dim, hidden_dim, critic_version):
        super().__init__()

        if critic_version == 'control_1_hidden':

            self.Q1 = util.mlp(critic_feed_feature_dim, hidden_dim, 1, 1)
            self.Q2 = util.mlp(critic_feed_feature_dim, hidden_dim, 1, 1)

            self.outputs = dict()
            self.apply(util.weight_init)


        elif critic_version == 'control_2_hidden':

            self.Q1 = util.mlp(critic_feed_feature_dim, hidden_dim, 1, 2)
            self.Q2 = util.mlp(critic_feed_feature_dim, hidden_dim, 1, 2)

            self.outputs = dict()
            self.apply(util.weight_init)


        elif critic_version == 'basic_sin':

            self.Q1 = nn.Sequential(

                      nn.Linear(critic_feed_feature_dim, hidden_dim),  # random feature
                      Sin(),
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.ELU(),
                      nn.Linear(hidden_dim, 1),

                      )

            self.Q2 = nn.Sequential(

                      nn.Linear(critic_feed_feature_dim, hidden_dim),  # random feature
                      Sin(),
                      nn.Linear(hidden_dim, hidden_dim),
                      nn.ELU(),
                      nn.Linear(hidden_dim, 1),

                      )

            self.outputs = dict()


        elif critic_version == 'elu_sin':

            self.Q1 = nn.Sequential(

                nn.Linear(critic_feed_feature_dim, hidden_dim),  # random feature
                SinElu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),

            )

            self.Q2 = nn.Sequential(

                nn.Linear(critic_feed_feature_dim, hidden_dim),  # random feature
                SinElu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),

            )

            self.outputs = dict()


        else:
            return






    def forward(self, critic_feed_feature):
        obs_action = critic_feed_feature
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


