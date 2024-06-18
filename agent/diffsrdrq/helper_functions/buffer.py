import collections

import numpy as np
import torch

Batch = collections.namedtuple(
    'Batch',
    ['state', 'action', 'reward', 'next_state', 'done']
)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device_num=None, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        if not device_num:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:" + str(device_num) if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return Batch(
            state=torch.FloatTensor(self.state[ind]).to(self.device),
            action=torch.FloatTensor(self.action[ind]).to(self.device),
            next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
            reward=torch.FloatTensor(self.reward[ind]).to(self.device),
            done=torch.FloatTensor(self.done[ind]).to(self.device),
        )

    def access(self, batch_size, right_index):
        # ind = np.random.randint(0, self.size, size=batch_size)
        ind = np.array(range(right_index - batch_size, right_index))

        return Batch(
            state=torch.FloatTensor(self.state[ind]).to(self.device),
            action=torch.FloatTensor(self.action[ind]).to(self.device),
            next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
            reward=torch.FloatTensor(self.reward[ind]).to(self.device),
            done=torch.FloatTensor(self.done[ind]).to(self.device),
        )
