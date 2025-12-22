import os

import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from spectralrl.algo.state import SAC, Ctrl_SAC, LVRep_SAC, Speder_SAC
from spectralrl.buffer.state import ReplayBuffer
from spectralrl.utils.logger import TensorboardLogger
from spectralrl.utils.utils import set_device, set_seed_everywhere


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # setup logger, seed and device
        self.logger = TensorboardLogger(
            "/".join([cfg.log_dir, cfg.algo.cls, cfg.name, cfg.task]),
            "_".join(["seed"+str(cfg.seed), cfg.name]),
            activate=not cfg.debug
        )
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        self.seed = set_seed_everywhere(cfg.seed)
        self.device = set_device(cfg.device)

        # setup envs
        self.train_env = gym.make(cfg.task)
        self.eval_env = gym.make(cfg.task)
        self.train_env.seed(cfg.seed)
        self.eval_env.seed(cfg.seed)
        self.max_episode_length = getattr(self.train_env, "_max_episode_steps")

        # setup buffer
        self.buffer = ReplayBuffer(
            obs_dim=self.train_env.observation_space.shape[0],
            action_dim=self.train_env.action_space.shape[0],
            max_size=cfg.buffer_size
        )

        # setup agent
        algo_cls = {
            "sac": SAC,
            "lvrep_sac": LVRep_SAC,
            "ctrl_sac": Ctrl_SAC,
            "speder_sac": Speder_SAC
        }.get(cfg.algo.cls)
        self.agent = algo_cls(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            cfg.algo,
            self.device
        )

        self.global_step = 0
        self.global_episode = 0

    def train(self):
        cfg = self.cfg

        ep_length, ep_return = 0, 0
        obs = self.train_env.reset()
        for t in trange(int(cfg.train_steps+1), desc="main"):
            if t < cfg.random_steps:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs, self.global_step, deterministic=False)

            next_obs, reward, terminal, info = self.train_env.step(action)
            ep_length += 1
            ep_return += reward
            timeout = info.get("TimeLimit.truncated", False) or \
                (self.max_episode_length is not None and ep_length >= self.max_episode_length)

            self.buffer.add(obs, action, next_obs, reward, terminal and not timeout)
            obs = next_obs

            if terminal or timeout:
                obs = self.train_env.reset()
                self.global_episode += 1
                self.logger.log_scalars("", {
                    "rollout/episode_return": ep_return,
                    "rollout/episode_length": ep_length
                }, step=self.global_step)
                ep_length = ep_return = 0

            if t < cfg.warmup_steps:
                train_metrics = {}
            else:
                train_metrics = self.agent.train_step(self.buffer, cfg.batch_size)

            if t % cfg.log_interval == 0:
                self.logger.log_scalars("", train_metrics, step=self.global_step)

            if t % cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.logger.log_scalars("eval", eval_metrics, step=self.global_step)

            self.global_step += 1

    def evaluate(self):
        self.agent.train(False)
        all_lengths = []
        all_returns = []
        for i_episode in range(self.cfg.eval_episodes):
            obs = self.eval_env.reset()
            ep_length = ep_return = 0
            terminal = False
            while not terminal:
                action = self.agent.select_action(obs, self.global_step, deterministic=True)
                obs, reward, terminal, info = self.eval_env.step(action)
                ep_return += reward
                ep_length += 1
            all_lengths.append(ep_length)
            all_returns.append(ep_return)
        all_lengths = np.asarray(all_lengths)
        all_returns = np.asarray(all_returns)
        self.agent.train(True)
        metrics = {
            "return_mean": all_returns.mean(),
            "return_std": all_returns.std(),
            "length_mean": all_lengths.mean()
        }
        return metrics


@hydra.main(version_base=None, config_path="./config/state", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
