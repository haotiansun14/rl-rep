# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os,sys
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import numpy as np
import torch
from dm_env import specs

import metaworld_env
import agent_utils as utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from drqv2 import DrQV2Agent
from mulv_config import config as cfg

torch.backends.cudnn.benchmark = True


class Workspace:
    def __init__(self,cfg):
        self.cfg = cfg
        self.work_dir = Path.cwd()
        self.work_dir = self.work_dir / 'agent'/ 'mulvdrq' /cfg.task_name / str(cfg.seed)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f'workspace: {self.work_dir}')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = DrQV2Agent(self.train_env.observation_spec().shape,self.train_env.action_spec().shape,self.cfg)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        if cfg.print:
            sys.stdout = utils.Logger(self.work_dir,sys.stdout)
            sys.stderr = utils.Logger(self.work_dir,sys.stderr)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        print('self.cfg.task_name:',self.cfg.task_name)
        self.train_env = metaworld_env.make(self.cfg.task_name, self.cfg.f_sta,
                                             self.cfg.a_re, self.cfg.seed)
        self.eval_env = metaworld_env.make(self.cfg.task_name, self.cfg.f_sta,
                                            self.cfg.a_re, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        print('data_specs:',data_specs)

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.buf_size,
            self.cfg.b_size, self.cfg.buf_workers,
            self.cfg.snap, self.cfg.nstep, self.cfg.disc)
        self._replay_iter = None


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.a_re

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward,total_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_ev)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            tmp_success = 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                    # print('action:',action)
                time_step = self.eval_env.step(action)
                total_reward += time_step.reward
                tmp_success += time_step.success
                step += 1
            total_success += (tmp_success >=1.0)

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.a_re / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('episode_success', total_success / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.a_re)
        seed_until_step = utils.Until(self.cfg.seed_f,
                                      self.cfg.a_re) # update network after seed_f
        eval_every_step = utils.Every(self.cfg.ev_every,
                                      self.cfg.a_re)

        episode_step, episode_reward, episode_success = 0, 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.a_re
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode_success', (episode_success>=1.)/1.) # True/1 =1, Flase/1 = 0
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0
                episode_success = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # autoencoder pretrian
            if self.global_frame == self.cfg.seed_f:
                for i in range(self.cfg.pre_step+1):
                    metrics = self.agent.update(self.replay_iter, self.global_step, pretrain=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.up_freq):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            episode_success += time_step.success
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1


def main():
    workspace = Workspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()