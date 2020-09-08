from bandit import Bandit
from distributions import trt_dist_list, num_obs
import numpy as np
from utils import ucb_value_naive, lcb_value_naive


def successive_elimination(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in successive_elimination
    algorithm"""
    
    # set all arms as active
    active = [1 for arm in range(num_arms)]
    for round in range(num_rounds):
        # pull active arms
        for arm in range(num_arms):
            if active[arm] == 1:
                bandit.pull_arm(arm)
            else:
                pass
        # calculate ucb and lcb
        ucb = ucb_value_naive(num_arms, num_rounds, bandit.arm_pull_tracker,
                        bandit.avg_reward_tracker)
        lcb = lcb_value_naive(num_arms, num_rounds, bandit.arm_pull_tracker,
                        bandit.avg_reward_tracker)
        
        # Deactivate all arms that satisfy condition ucb(a) < lcb(a')
        for arm in range(num_arms):
            # find max_lcb of all other arms
            lcb_other = [lcb[i] for i in range(len(lcb)) if i != arm]
            # if this condition then deactivate arm
            if ucb[arm] < np.amax(lcb_other):
                active[arm] = 0
    
    return bandit
