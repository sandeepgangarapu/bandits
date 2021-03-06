from bandits.bandit import Bandit
from bandits.distributions import trt_dist_list, num_obs
import numpy as np
import random


def explore_first(bandit, num_rounds, explore_percentage, num_arms):
    """Function that reproduces the steps involved in explore_first
    algorithm"""
    num_explore = int((num_rounds * explore_percentage) / 100)
    num_explore_each_arm = int(num_explore / num_arms)
    num_exploit = num_rounds - num_explore
    
    # Explore all arms first:
    for round in range(num_explore_each_arm):
        for arm in range(num_arms):
            bandit.pull_arm(arm)
    
    # Exploit max reward arm
    max_arm = bandit.max_reward_arm
    for round in range(num_exploit):
        bandit.pull_arm(max_arm)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms_ex = 4
    explore_percentage = 10
    num_rounds = num_obs
    trt_dist_lis_ex = trt_dist_list[:num_arms_ex]
    explore_bandit = Bandit(name='explore_first',
                            num_arms=num_arms_ex,
                            trt_dist_list=trt_dist_lis_ex)
    explore_first(bandit=explore_bandit,
                 num_rounds=num_rounds,
                 explore_percentage=explore_percentage,
                 num_arms=num_arms_ex)