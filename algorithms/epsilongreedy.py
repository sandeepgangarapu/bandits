from bandits.bandit import Bandit
from bandits.distributions import trt_dist_list, num_obs
import numpy as np
import random


def epsilon_greedy(epsilon, bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in epsilon greedy
    algorithm"""
    
    for round in range(num_rounds):
        flip = random.random()
        if flip < epsilon:
            # if random flip is less than threshold, we explore
            # choose an arm that does not have max_reward randomly
            arm_other = np.random.choice([i for i in range(num_arms) if i!=
                                          bandit.max_reward_arm])
            bandit.pull_arm(arm_other)
        else:
            # if random flip is greater than threshold, we exploit
            # pull the arm with max reward so far
            bandit.pull_arm(bandit.max_reward_arm)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms_ep = 4
    epsilon = 0.5
    num_rounds = num_obs
    trt_dist_lis_ep = trt_dist_list[:num_arms_ep]
    epsilon_bandit = Bandit(name='epsilon_greedy',
                            num_arms=num_arms_ep,
                            trt_dist_list=trt_dist_lis_ep)
    epsilon_greedy(epsilon=epsilon, bandit=epsilon_bandit,
                  num_rounds=num_rounds, num_arms=num_arms_ep)