from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, num_obs
import numpy as np
from bandits.utils import ucb_value_naive


def ucb(bandit, num_rounds):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""
    print("---------------Running UCB ---------------")
    # choose each action once:
    for arm in range(bandit.num_arms):
        bandit.pull_arm(arm)
    
    ucb_rounds = num_rounds - bandit.num_arms
    for round in range(ucb_rounds):
        # find UCB for all arms
        ucb_round = ucb_value_naive(bandit.num_arms, ucb_rounds, bandit.arm_pull_tracker,
                                    bandit.avg_reward_tracker)
        # find arm with max UCB
        arm_max_ucb = np.argmax(ucb_round)
        # Pull the arm with max ucb
        bandit.pull_arm(arm_max_ucb)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = num_obs
    trt_dist_lis = trt_dist_list[:num_arms]
    ucb_bandit = Bandit(name='ucb_naive',
                            num_arms=num_arms,
                            trt_dist_list=trt_dist_lis)
    ucb(bandit=ucb_bandit,
              num_rounds=num_rounds)