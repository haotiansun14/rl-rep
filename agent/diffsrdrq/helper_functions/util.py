import time
import copy
import numpy as np
from torch import nn
import torch
import re


def unpack_batch(batch):
    return batch.state, batch.action, batch.next_state, batch.reward, batch.done


class Timer:

    def __init__(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def reset(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def set_step(self, step):
        self._step = step
        self._step_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time

    def steps_per_sec(self, step):
        sps = (step - self._step) / (time.time() - self._step_time)
        self._step = step
        self._step_time = time.time()
        return sps


def eval_policy(policy, eval_env, eval_episodes=10):
    """
    Eval a policy
    """
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target

def convert_to_tensor(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return torch.from_numpy(obj).to(device)


def generate_alphas_and_alphabars(param1=1e-4, param2=2e-2, num=20, noise_schedule="linear"):
    if noise_schedule == "linear":
        betas = np.linspace(param1, param2, num)
    elif noise_schedule == "vp":
        t = np.arange(1, num+1)
        T = num
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
    else:
        raise NotImplementedError
    alphas = 1-betas
    alphabars = np.cumprod(alphas, axis=0)
    return torch.as_tensor(betas, dtype=torch.float32), \
           torch.as_tensor(alphas, dtype=torch.float32), \
           torch.as_tensor(alphabars, dtype=torch.float32)

def sync_target(src, tgt, tau):
    for o, n in zip(tgt.parameters(), src.parameters()):
        o.data.copy_(o.data * (1.0 - tau) + n.data * tau)

def setup_schedule(schdl):
    match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
    init, final, duration = [float(g) for g in match.groups()]
    def fn(step):
        mix = np.clip(step / duration, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final
    return fn

if __name__ == '__main__':

    q_test = mlp(2048, 1024, 1, 2)
    print(q_test)
    # summary(q_test, (256, 2048), verbose=2)
