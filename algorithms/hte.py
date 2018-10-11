from bandits.bandit import Bandit
from bandits.distributions import trt_dist_list, num_obs
import numpy as np
import random


def always_explore(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in explore_first
    algorithm"""
    num_explore_each_arm = int(num_rounds / num_arms)
    for round in range(num_explore_each_arm):
        for arm in range(num_arms):
            bandit.pull_arm(arm)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms_ex = 4
    num_rounds = num_obs
    trt_dist_lis_ex = trt_dist_list[:num_arms_ex]
    always_explore_bandit = Bandit(name='always_explore',
                            num_arms=num_arms_ex,
                            trt_dist_list=trt_dist_lis_ex)
    always_explore(bandit=always_explore_bandit,
                  num_rounds=num_rounds,
                  num_arms=num_arms_ex)