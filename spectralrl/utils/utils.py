import copy
import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed_everywhere(seed: Optional[Union[str, int]] = None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def set_device(id: Optional[Union[str, int, torch.device]] = None) -> torch.device:
    """Selects the device according to given ID.

    :param id: ID of the device to select, its value can be

        - `None`: select the most free gpu (if cuda is available) or cpu (otherwise).
        - `str`: if `"cpu"`, then cpu will be selected; if `"x"` or `"cuda:x"`, then the `x`-th gpu will be selected.
        - int: equals to `"x"`.
        - :class:``~torch.device``, then select that device.
    """

    if isinstance(id, torch.device):
        return id
    if not torch.cuda.is_available() or id == "cpu":
        return torch.device("cpu")
    if id is None:
        return select_free_cuda()
    try:
        id = int(id)
    except ValueError:
        try:
            id = int(id.split(":")[-1])
        except Exception:
            raise ValueError("Invalid cuda ID format: {}".format(id))

    if id < 0:
        return torch.device("cpu")
    else:
        ret =  torch.device(f"cuda:{id}")
        return ret

def select_free_cuda():
    def get_volume(t):
        return np.asarray([int(i.split()[2]) for i in t])

    try:
        import numpy as np
        cmd1 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total")
        total = cmd1.readlines()
        cmd1.close()

        cmd2 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Reserved")
        reserved = cmd2.readlines()
        cmd2.close()

        cmd3 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Used")
        used = cmd3.readlines()
        cmd3.close()

        def get_volume(t):
            return np.asarray([int(i.split()[2]) for i in t])

        total, used, reserved = get_volume(total), get_volume(used), get_volume(reserved)
        return torch.device("cuda:"+str(np.argmax(total-used-reserved)))
    except Exception as e:
        cmd4 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free")
        free = cmd4.readlines()
        cmd4.close()

        free = get_volume(free)
        return torch.device("cuda:"+str(np.argmax(free)))

def make_target(m: nn.Module) -> nn.Module:
    target = copy.deepcopy(m)
    target.requires_grad_(False)
    target.eval()
    return target

def sync_target(src, tgt, tau):
    for o, n in zip(tgt.parameters(), src.parameters()):
        o.data.copy_(o.data * (1.0 - tau) + n.data * tau)

def convert_to_tensor(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return torch.from_numpy(obj).to(device)

def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")
