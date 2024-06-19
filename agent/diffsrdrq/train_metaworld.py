import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
from dm_env import specs
from latent_diff_sr import LatentDiffSRDrQv2
from drqv2 import DrQv2
from helper_functions.efficient_buffer import EfficientReplayBuffer
from helper_functions.drqv2_buffer import(
    ReplayBufferStorage,
    make_replay_loader,
)
from helper_functions.util import Timer
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from UtilsRL.misc.decorator import profile
from pathlib import Path
from tqdm import trange
import wandb
from helper_functions.video import VideoRecorder

class Workspace():
    def __init__(self, args):
        self.args = args
        
        # create logger
        log_dir = "/".join([args["log_dir"], args["name"], args["task"]])
        exp_name = "_".join(["seed"+str(args.seed)])
        self.logger = CompositeLogger(log_dir=log_dir, name=exp_name,
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {**args["wandb"], "config": args, "settings": wandb.Settings(_disable_stats=True)}, 
                "CsvLogger": {**args["csv"]}
            }, activate=not args["debug"])
        self.logger.log_config(args)

        # create envs
        domain, task = args.task.split("_")
        self.domain = domain
        if domain == "metaworld":
            import env.metaworld_env as metaworld_env
            self.train_env = metaworld_env.make(task, args.frame_stack, args.action_repeat, args.env_seed)
            self.eval_env = metaworld_env.make(task, args.frame_stack, args.action_repeat, args.env_seed)
        elif domain == "dmc":
            import env.dmc_env as dmc_env
            self.train_env = dmc_env.make(task, args.frame_stack, args.action_repeat, args.env_seed)
            self.eval_env = dmc_env.make(task, args.frame_stack, args.action_repeat, args.env_seed)
        else:
            raise ValueError(f"Unrecognized domain: {self.domain}.")
            
        # create buffer
        self.work_dir = Path.cwd()
        self.work_dir = self.work_dir / ".cache" / args.task / str(args.seed)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.buffer_type = args.buffer_type
        data_specs=(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, "reward"),
            specs.Array((1, ), np.float32, "discount")
        ) 
        if self.buffer_type == "numpy":
            self.replay_buffer = EfficientReplayBuffer(
                buffer_size=args.buffer_size, 
                batch_size=args.batch_size, 
                nstep=args.nstep, 
                discount=args.discount, 
                frame_stack=args.frame_stack, 
                data_specs=data_specs,
            )
            self._replay_iter = self.replay_buffer
        elif self.buffer_type == "storage":
            self.replay_buffer = ReplayBufferStorage(
                data_specs=data_specs, 
                replay_dir=self.work_dir/ "buffer"
            )
            self.replay_loader = make_replay_loader(
                self.work_dir / "buffer", args.buffer_size, args.batch_size,
                args.buffer_workers, args.buffer_snap, args.nstep, args.discount
            )
            self._replay_iter = None
        else:
            raise ValueError(f"Unknown buffer type: {args.buffer_type}")

        # create agent
        if args.algo == "drqv2":
            self.agent = DrQv2(
                self.train_env.observation_spec(), 
                self.train_env.action_spec(),
                args
            )
        elif args.algo == "latent_diff_sr":
            self.agent = LatentDiffSRDrQv2(
                self.train_env.observation_spec(), 
                self.train_env.action_spec(), 
                args
            )
        self._global_step = 0
        self._global_episode = 0
        self.timer = Timer()
        self.video_recorder = VideoRecorder(self.work_dir if args.save_video else None)
        
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self._global_step * self.args.action_repeat
    
    def eval(self):
        all_lengths = []
        all_returns = []
        all_success = []
        self.agent.train(False)
        for i_episode in range(self.args.eval_episodes):
            time_step = self.eval_env.reset()
            length = ret = success = 0
            self.video_recorder.init(self.eval_env, enabled=(i_episode==0))
            while not time_step.last():
                action = self.agent.select_action(time_step.observation, self.global_step, deterministic=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                ret += time_step.reward
                length += 1
                if hasattr(time_step, "success"):
                    success += float(time_step.success)
            self.video_recorder.save(f"eval_{self.global_frame}.mp4")
            all_lengths.append(length)
            all_returns.append(ret)
            all_success.append(float(success>=1.0))
        all_lengths = np.asarray(all_lengths)
        all_returns = np.asarray(all_returns)
        all_success = np.asarray(all_success, dtype=np.float32)
        self.agent.train(True)
        metrics = {
            "return_mean": all_returns.mean(), 
            "return_std": all_returns.std(), 
            "length_mean": all_lengths.mean(), 
            "success_mean": all_success.mean()
        }
        # save another sampling video
        time_step = self.eval_env.reset()
        self.video_recorder.init(self.eval_env, enabled=True)
        while not time_step.last():
            action = self.agent.select_action(time_step.observation, self.global_step, deterministic=False)
            time_step = self.eval_env.step(action)
            self.video_recorder.record(self.eval_env)
        self.video_recorder.save(f"train_{self.global_frame}.mp4")
        
        # do reconstruction validation
        if self.global_step > 0:
            agent_metrics, grid = self.agent.evaluate(self.replay_iter)
            metrics.update(agent_metrics)
            if grid is not None:
                self.logger.log_image("info/reconstruction", grid, step=self.global_frame)
        return metrics
    
    def train(self):
        args = self.args
        episode_step, episode_return, episode_success = 0, 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        for i_frame in trange(args.num_train_frames // args.action_repeat + 1, desc="main"):
            if time_step.last():
                self._global_episode += 1
                episode_frame = episode_step * args.action_repeat
                self.logger.log_scalars("", {
                    "rollout/return": episode_return,
                    "rollout/success": (episode_success >= 1.0)*1.0 if self.domain == "metaworld" else 0.0,
                    "rollout/episode_length": episode_frame
                }, step=self.global_frame)

                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                episode_step = episode_return = episode_success = 0

            if self.global_frame < args.warmup_frames:
                sample = self.train_env.action_spec().generate_value()
                action = np.random.uniform(low=-1, high=1, size=sample.shape)
                action = action.astype(sample.dtype)
                train_metrics = {}
            else:
                if self.global_frame == args.warmup_frames and args.pretrain_steps > 0:
                    for i_pretrain in trange(args.pretrain_steps, desc="pretrain"):
                        pretrain_metrics = self.agent.pretrain_step(self.replay_iter, step=i_pretrain)
                        if i_pretrain % 1000 == 0:
                            gen_metrics, grid = self.agent.evaluate(self.replay_iter)
                            pretrain_metrics.update(gen_metrics)
                            self.logger.log_scalars("pretrain", pretrain_metrics, step=i_pretrain)
                            self.logger.log_image("pretrain/recon", grid, step=i_pretrain)
                    
                with torch.no_grad():
                    action = self.agent.select_action(time_step.observation, self.global_step, deterministic=False)

                for i_update in range(args.update_freq):
                    train_metrics = self.agent.train_step(self.replay_iter, self.global_step)
            
            if self.global_frame % args.log_frames == 0:
                self.logger.log_scalars("", train_metrics, step=self.global_frame)

            if self.global_frame % args.eval_frames == 0:
                eval_metrics = self.eval()
                self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
                self.logger.info(eval_metrics)

            time_step = self.train_env.step(action)
            episode_return += time_step.reward
            if self.domain == "metaworld":
                episode_success += time_step.success
            self.replay_buffer.add(time_step)
            episode_step += 1
            self._global_step += 1


if __name__ == "__main__":
    args = parse_args()
    setup(args)
    print(f"Using device: {args.device}")

    if args.wandb.activate: 
        wandb.login(key=os.environ["wandb_api_key"])
    workspace = Workspace(args)
    workspace.train()
