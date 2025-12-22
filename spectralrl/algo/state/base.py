from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseStateAlgorithm(ABC):

    def train(self, *args, **kwargs):
        return

    @abstractmethod
    def select_action(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError
