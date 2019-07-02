from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, num_obs
import numpy as np
import random


def thompson_bandit(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    
    
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms_ep = 4
    num_rounds = num_obs
    trt_dist_lis_ep = trt_dist_list[:num_arms_ep]
    thompson_bandit = Bandit(name='thompson_sampling',
                            num_arms=num_arms_ep,
                            trt_dist_list=trt_dist_lis_ep)
    thompson_bandit(bandit=thompson_bandit, num_rounds=num_rounds,
                    num_arms=num_arms_ep)