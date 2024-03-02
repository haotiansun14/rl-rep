# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import agent_utils as utils
import itertools
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import vae
import copy

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    # encoder for the 3 framestack images
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3 # todo: why 3?
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        # (size - (kernel_size - 1) - 1) // stride + 1

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5 # normalize to [-0.5, 0.5]
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class PredictEncoder(nn.Module):
    # encoder for the predicted image
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        # (size - (kernel_size - 1) - 1) // stride + 1

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5 # normalize to [-0.5, 0.5]
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Decoder(nn.Module):
    def __init__(self, obs_shape = (32,35,35)):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.deconvnet = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 37, 37])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 39, 39])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1), # torch.Size([256, 32, 41, 41])
                                     nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=2), # torch.Size([256, 32, 83, 83])
                                     nn.ReLU(), nn.Conv2d(32, 3, 2, stride=1,padding=1), # torch.Size([256, 3, 84, 84])
                                    )
        # todo: more complicated net?
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], 32, 35, 35)
        h = self.deconvnet(obs) # [0,+inf] because of ReLU
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    """
    Critic with random fourier features
    """

    def __init__(
            self,
            feature_dim,
            c_noise,
            q_activ,
            num_noise=20,
            hidden_dim=256,
            device=torch.device('cpu'),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_noise = num_noise
        self.c_noise = c_noise
        self.device = device
        if q_activ == 'relu':
            self.q_activ = F.relu
        elif q_activ == 'elu':
            self.q_activ = F.elu
        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, mean, log_std):
        """
        """
        std = log_std.exp()
        batch_size, d = mean.shape

        x = mean[:, None, :] + std[:, None, :] * torch.randn([self.num_noise, self.feature_dim], requires_grad=False, device=self.device) * self.c_noise
        x = x.reshape(-1, d)

        q1 = self.q_activ(self.l1(x))  # F.relu(self.l1(x))
        q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = self.q_activ(self.l2(q1))  # F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = self.q_activ(self.l4(x))  # F.relu(self.l4(x))
        q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = self.q_activ(self.l5(q2))  # F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.c_targ_tau = cfg.c_targ_tau
        self.up_every = cfg.up_every
        self.q_up_n = cfg.q_up_n
        self.use_tb = cfg.use_tb
        self.num_expl_steps = cfg.num_expl_steps
        self.stddev_schedule = cfg.stddev_schedule
        self.stddev_clip = cfg.stddev_clip

        self.lat_feat_dim = cfg.lat_feat_dim
        self.use_feature_target = True
        self.vae_w = cfg.vae_w
        self.mse_w = cfg.mse_w
        self.aug = cfg.aug
        self.pre_aug = cfg.pre_aug # predict pic after aug
        self.back_q2feat = cfg.back_q2feat
        self.c_noise = cfg.c_noise
        self.l2_norm = cfg.l2_norm
        self.tanh = cfg.tanh
        self.both_q = cfg.both_q
        self.q_activ = cfg.q_activ
        self.q_loss = cfg.q_loss

        # models
        self.encoder = Encoder(obs_shape).to(self.device) # encode the 3 framestack images
        self.encoder_target = Encoder(obs_shape).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.decoder = Decoder().to(self.device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, cfg.feat_dim, cfg.hid_dim).to(cfg.device)

        self.critic = Critic(feature_dim=cfg.feat_dim,c_noise=self.c_noise,q_activ=self.q_activ,hidden_dim=cfg.hid_dim,device=cfg.device).to(cfg.device)
        self.critic_target = Critic(feature_dim=cfg.feat_dim,c_noise=self.c_noise,q_activ=self.q_activ,hidden_dim=cfg.hid_dim,device=cfg.device).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.predict_encoder = PredictEncoder(obs_shape).to(cfg.device) # encode 1 image
        self.feat_encoder = vae.Encoder(state_dim=self.encoder.repr_dim,action_dim=action_shape[0],tanh=self.tanh,feature_dim=cfg.feat_dim).to(cfg.device)
        self.feat_decoder = vae.Decoder(state_dim=self.encoder.repr_dim,feature_dim=cfg.feat_dim,hidden_dim=cfg.hid_dim).to(cfg.device)
        self.feat_f = vae.GaussianFeature(state_dim=self.encoder.repr_dim,action_dim=action_shape[0],tanh=self.tanh,feature_dim=cfg.feat_dim).to(cfg.device)
        if self.use_feature_target:
            self.feat_f_target = copy.deepcopy(self.feat_f)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)

        self.predict_encoder_opt = torch.optim.Adam(self.predict_encoder.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.feat_encoder_opt = torch.optim.Adam(self.feat_encoder.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.feat_decoder_opt = torch.optim.Adam(self.feat_decoder.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)
        self.feat_f_opt = torch.optim.Adam(self.feat_f.parameters(), lr=cfg.lr,weight_decay=self.l2_norm)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.decoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.predict_encoder.train(training)
        self.feat_encoder.train(training)
        self.feat_decoder.train(training)
        self.feat_f.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        obs = obs.view(obs.shape[0], -1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        mean, log_std = self.feat_f(obs, action)
        Q1, Q2 = self.critic(mean, log_std)
        if self.both_q:
            Q = torch.concat([Q1,Q2],dim=1)
        else:
            Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step,pretrain=False,save_fig_index=None):
        metrics = dict()

        if step % self.up_every != 0:
            return metrics

        for i in range(self.q_up_n):
            batch = next(replay_iter)
            img, action, reward, discount, next_img, img_step1 = utils.to_torch(batch, self.device)

            if self.aug:
                if self.pre_aug:
                    img_w_pred = torch.cat([img, img_step1[:,-3:,:,:]], dim=1)
                    img_w_pred = self.aug(img_w_pred.float())
                    img = img_w_pred[:,:-3,:,:]
                    img_step1 = img_w_pred[:,-3:,:,:]
                else:
                    img = self.aug(img.float())
                    img_step1 = img_step1[:,-3:,:,:] # no aug for image prediction
                next_img = self.aug(next_img.float())
            else:
                img = img.float()
                img_step1 = img_step1[:,-3:,:,:]
                next_img = next_img.float() # todo:test without using aug again
            # encode
            state = self.encoder(img) # vector
            state_step1 = self.predict_encoder(img_step1)

            # ML loss
            z = self.feat_encoder.sample(state, action, state_step1)
            x, r = self.feat_decoder(z)
            pred_img_step1 = self.decoder(x)

            img_step1 = img_step1 / 255.0 - 0.5  # normalize to [-0.5, 0.5]
            s_loss = F.l1_loss(pred_img_step1, img_step1) * 10.
            r_loss = F.mse_loss(r, reward)
            ml_loss = r_loss + s_loss

            # KL loss
            mean1, log_std1 = self.feat_encoder(state, action, state_step1)
            mean2, log_std2 = self.feat_f(state, action)
            var1 = (2 * log_std1).exp()
            var2 = (2 * log_std2).exp()
            kl_loss = log_std2 - log_std1 + 0.5 * (var1 + (mean1 - mean2) ** 2) / var2 - 0.5
            kl_loss = kl_loss.mean()

            autoencoder_loss = ml_loss * self.mse_w + kl_loss
            autoencoder_loss = autoencoder_loss * self.vae_w

            if pretrain:
                self.encoder_opt.zero_grad(set_to_none=True)
                self.decoder_opt.zero_grad(set_to_none=True)
                self.predict_encoder_opt.zero_grad(set_to_none=True)
                self.feat_encoder_opt.zero_grad(set_to_none=True)
                self.feat_decoder_opt.zero_grad(set_to_none=True)
                self.feat_f_opt.zero_grad(set_to_none=True)

                autoencoder_loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                self.predict_encoder_opt.step()
                self.feat_encoder_opt.step()
                self.feat_decoder_opt.step()
                self.feat_f_opt.step()

                # update target network
                self.encoder_target.load_state_dict(self.encoder.state_dict())
                if self.use_feature_target:
                    self.feat_f_target.load_state_dict(self.feat_f.state_dict())

            else:
                if self.back_q2feat:
                    with torch.no_grad():
                        next_state = self.encoder_target(next_img)

                        stddev = utils.schedule(self.stddev_schedule, step)
                        dist = self.actor(next_state, stddev)
                        next_action = dist.sample(clip=self.stddev_clip)
                        if self.use_feature_target:
                            next_mean, next_log_std = self.feat_f_target(next_state, next_action)
                        else:
                            next_mean, next_log_std = self.feat_f(next_state, next_action)
                        target_Q1, target_Q2 = self.critic_target(next_mean, next_log_std)
                        target_V = torch.min(target_Q1, target_Q2)
                        target_Q = reward + (discount * target_V)

                    mean, log_std = self.feat_f(state, action) # state use feat_f and has gradients backward
                else:
                    with torch.no_grad():
                        # next_state = self.encoder(next_img)
                        next_state = self.encoder_target(next_img)

                        stddev = utils.schedule(self.stddev_schedule, step)
                        dist = self.actor(next_state, stddev)
                        next_action = dist.sample(clip=self.stddev_clip)
                        if self.use_feature_target:
                            mean, log_std = self.feat_f_target(state, action)
                            next_mean, next_log_std = self.feat_f_target(next_state, next_action)
                        else:
                            mean, log_std = self.feat_f(state, action)
                            next_mean, next_log_std = self.feat_f(next_state, next_action)
                        target_Q1, target_Q2 = self.critic_target(next_mean, next_log_std)
                        target_V = torch.min(target_Q1, target_Q2)
                        target_Q = reward + (discount * target_V)

                Q1, Q2 = self.critic(mean, log_std)
                if self.q_loss == 'mse':
                    critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
                elif self.q_loss == 'huber':
                    critic_loss = F.smooth_l1_loss(Q1, target_Q) + F.smooth_l1_loss(Q2, target_Q)

                loss = critic_loss + autoencoder_loss

                self.encoder_opt.zero_grad(set_to_none=True)
                self.decoder_opt.zero_grad(set_to_none=True)
                self.predict_encoder_opt.zero_grad(set_to_none=True)
                self.feat_encoder_opt.zero_grad(set_to_none=True)
                self.feat_decoder_opt.zero_grad(set_to_none=True)
                self.feat_f_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)

                loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                self.predict_encoder_opt.step()
                self.feat_encoder_opt.step()
                self.feat_decoder_opt.step()

                self.feat_f_opt.step()
                self.critic_opt.step()

        # update actor
        metrics.update(self.update_actor(state.detach(), step))

        if self.c_targ_tau < 1: # soft update
            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,self.c_targ_tau)
            utils.soft_update_params(self.encoder, self.encoder_target,self.c_targ_tau)
            # Update the feature network if needed
            if self.use_feature_target:
                utils.soft_update_params(self.feat_f, self.feat_f_target,self.c_targ_tau)
        else: # hard update
            if step % self.c_targ_tau == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.encoder_target.load_state_dict(self.encoder.state_dict())
                if self.use_feature_target:
                    self.feat_f_target.load_state_dict(self.feat_f.state_dict())

        return metrics