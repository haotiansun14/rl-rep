import numpy as np
import torch
import torch.nn as nn


class DummyNormalizer(nn.Module):
    def __init__(self, epsilon=1e-8, shape=(), dtype=torch.float32):
        super().__init__()

    def update(self, x):
        return

    def normalize(self, x):
        return x


class RunningMeanStdNormalizer(nn.Module):
    def __init__(self, epsilon=1e-8, shape=(), dtype=torch.float32):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=dtype))
        self.register_buffer("count", torch.tensor(epsilon, dtype=dtype))
        self.epsilon = 1e-8

    def update(self, x):
        assert x.shape[0] == 1

        self.count += 1
        delta1 = x.squeeze() - self.mean
        self.mean += delta1 / self.count
        delta2 = x.squeeze().pow(2) - self.mean_square
        self.mean_square += delta2 / self.count

    def normalize(self, x):
        var = self.mean_square - self.mean.pow(2)
        std = torch.sqrt(var + self.epsilon)
        std = torch.clip(std, min=1e-4)
        return (x - self.mean) / std
