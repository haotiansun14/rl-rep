# This file is used to define the configuration of the experiment.

class Dict(dict):
  # https://www.jb51.net/article/186264.htm
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def dict_to_object(dictObj):
  if not isinstance(dictObj, dict):
    return dictObj
  inst=Dict()
  for k,v in dictObj.items():
    inst[k] = dict_to_object(v)
  return inst

config = {'task_name':'metaworld_hand_insert',
          'num_train_frames': 1001000,
          'stddev_schedule': 'linear(1.0,0.1,500000)',
          # task settings
          'f_sta': 3,  # frame_stack
          'a_re': 2,  # action_repeat
          'disc': 0.99,  # discount factor
          # train settings
          'seed_f': 4000,  # number of random frames
          # autoencoder
          'pre_step': 0,  # 10000 # number of pretrain steps
          'vae_w': 0.5,  # 0.5 # weight of vae loss weight
          'mse_w': 1.0,  # weight of mse loss weight
          'aug': True,  # use data augmentation
          'pre_aug': False,  # use data augmentation for prediction
          # eval
          'ev_every': 10000,  # evaluation frequency
          'num_ev': 10,  # number of evaluation episodes
          # snapshot
          'snap': False,  # save snapshot
          # replay buffer
          'buf_size': 1000000,  # replay buffer size
          'buf_workers': 4,  # number of replay buffer workers
          'nstep': 3,  # nstep
          'b_size': 256, # batch size
          # misc
          'seed': 1,  # random seed
          'device': 'cuda',  # cuda or cpu
          'use_tb': False,  # use tensorboard
          # experiment
          'experiment': 'exp',  # experiment name
          # agent
          'lr': 1e-4,  # learning rate
          'c_targ_tau': 0.01,  # critic target tau
          'up_freq': 1,  # update frequency
          'q_up_n': 1,  # q update number
          'feat_dim': 100,  # feature dim
          'hid_dim': 1024, # hidden dim
          'lat_feat_dim': 512,  # latent feature dim
          'back_q2feat': True,  # backward q to feature
          'c_noise': 0.1,  # 1.0 # critic noise scale
          'print': False,  # log print
          'l2_norm': 0.0,  # l2 norm
          'tanh': True,  # use tanh
          'both_q': False,  # use both q to update actor
          'q_activ': 'relu', # q activation, relu, elu
          'q_loss': 'huber',  # q loss, mse, huber
          'up_every': 2,  # update every steps
          'num_expl_steps': 2000,  # number of exploration steps
          'stddev_clip': 0.3,  # stddev clip
          }

config = dict_to_object(config)

