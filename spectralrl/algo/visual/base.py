from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVisualAlgorithm(ABC):

    def train(self, *args, **kwargs):
        return

    def evaluate(self, *args, **kwargs):
        return {}, None

    @abstractmethod
    def select_action(self, *args, **kwargs):
        raise NotImplementedError

    def pretrain_step(self, *args, **kwargs):
        return {}

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError
