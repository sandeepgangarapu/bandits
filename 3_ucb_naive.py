from bandits.bandit import Bandit
from bandits.distributions import trt_dist_list, num_obs
import numpy as np
import random
from bandits.utils import ucb_naive


def ucb_naive(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""
    
    # choose each action once:
    for arm in range(num_arms):
        bandit.pull_arm(arm)
    
    ucb_rounds = num_rounds - num_arms
    for round in range(ucb_rounds):
        # find UCB for all arms
        ucb_round = ucb_naive(num_arms, ucb_rounds, bandit.arm_pull_tracker,
                        bandit.avg_reward_tracker)
        # find arm with max UCB
        arm_max_ucb = np.argmax(ucb_round)
        # Pull the arm with max ucb
        bandit.pull_arm(arm_max_ucb)

    print(bandit.avg_reward_tracker)
    print(bandit.arm_pull_tracker)
    pass


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = num_obs
    trt_dist_lis = trt_dist_list[:num_arms]
    ucb_bandit = Bandit(num_arms=num_arms,
                            trt_dist_list=trt_dist_lis)
    ucb_naive(bandit=ucb_bandit,
              num_rounds=num_rounds,
              num_arms=num_arms)