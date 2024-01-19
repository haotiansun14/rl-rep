# Overview
This repo is dedicated to exploring the field of Representation Learning (RepL) with a specific focus on Reinforcement Learning (RL) and Causal Inference. Our goal is to build a comprehensive resource that integrates our latest research and practical implementations.

## Representation-based Reinforcement Learning
This repo contains implementations for RL with:
- Latent Variable Representations (LV), as outlined in [1].
- Contrastive Representations (CTRL), as described in [2].

### Directory
- `agent` hosts implementation files for various agents, including the Soft Actor-Critic baseline (`sac`), SAC with Latent Variable (`vlsac`), and SAC with Contrastive Representations (`ctrlsac`).
- `networks` contains base implementations for critics, policy networks, variational autoencoders (VAE), and more.
- `utils` comprises replay buffers and several auxiliary functions.

### Run
Execute the `main.py` script with your preferred arguments, such as `--alg` for algorithm type, `--env` for environment, and so on.

Example usage: `python main.py --alg vlsac --env HalfCheetah-v3`.

### References
[1] [Ren, Tongzheng, Chenjun Xiao, Tianjun Zhang, Na Li, Zhaoran Wang, Sujay Sanghavi, Dale Schuurmans, and Bo Dai. "Latent variable representation for reinforcement learning." arXiv preprint arXiv:2212.08765 (2022).](https://arxiv.org/abs/2212.08765)

[2] [Zhang, Tianjun, Tongzheng Ren, Mengjiao Yang, Joseph Gonzalez, Dale Schuurmans, and Bo Dai. "Making linear mdps practical via contrastive representation learning." In International Conference on Machine Learning, pp. 26447-26466. PMLR, 2022.](https://arxiv.org/abs/2207.07150)

## Spectral Representation for Causal Inference
*The code will be available soon.*
