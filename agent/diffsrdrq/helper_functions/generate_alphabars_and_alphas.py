import numpy as np
import torch
from collections import deque


def generate_alphabars_and_alphas(lowest_noise_alpha=0.99, highest_noise_alpha=0.01, num_alphas=15):


    def to_tensor(arr):
        # convert arr to tensor
        return torch.tensor(arr).float()

    def get_array_from_cumulative_prod_of_array(alphabars):

        result = deque()
        for k in range(len(alphabars) - 1, -1, -1):
            if k == 0:
                val = alphabars[k]
            else:
                val = np.true_divide(   alphabars[k] , alphabars[k - 1] )
            result.appendleft(val)
        return np.array(result)


    alphabars =  np.exp(  np.flip(  np.linspace(np.log(highest_noise_alpha), np.log(lowest_noise_alpha),
                           num_alphas)   )  ) #FIXME: The flip needs to go away, but this is the correct noise scheduel

    alphas = get_array_from_cumulative_prod_of_array(alphabars)

    return to_tensor(alphabars), to_tensor(alphas)



