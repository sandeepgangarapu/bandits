from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, num_obs
import numpy as np
import random


def epsilon_greedy(bandit, num_rounds, epsilon):
    """Function that reproduces the steps involved in epsilon greedy
    algorithm"""
    
    for round in range(num_rounds):
        flip = random.random()
        if flip < epsilon:
            # if random flip is less than threshold, we explore
            # choose an arm that does not have max_reward randomly
            arm_other = np.random.choice([i for i in range(bandit.num_arms) if i!=
                                          bandit.max_reward_arm])
            bandit.pull_arm(arm_other)
        else:
            # if random flip is greater than threshold, we exploit
            # pull the arm with max reward so far
            bandit.pull_arm(bandit.max_reward_arm)
    return bandit

