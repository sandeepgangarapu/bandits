from bandits.bandit import Bandit
from bandits.distributions import trt_dist_list, num_obs
import numpy as np
from bandits.utils import ucb_value_naive
from bandits.peeking.always_valid_p import always_valid_p


def ucb_peek(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""

    # choose each action once:
    for arm in range(num_arms):
        bandit.pull_arm(arm)
    
    ucb_rounds = num_rounds - num_arms
    
    p = [[1] for i in range(num_arms)]
    for round in range(ucb_rounds):
        # find UCB for all arms
        ucb_round = ucb_value_naive(num_arms, ucb_rounds,
                                    bandit.arm_pull_tracker,
                                    bandit.avg_reward_tracker)
        # sort arms based on ucb
        arm_sort_ucb = np.argsort(ucb_round)
        # descending order
        arm_sort_ucb = arm_sort_ucb[::-1]
        # check p_value
        p_int = []
        for arm in arm_sort_ucb:
            p_peek = always_valid_p(theta0=0, tau=1,
                                    X=bandit.arm_reward_tracker[0],
                                    Y=bandit.arm_reward_tracker[arm])
            p_min = min(p[arm][-1], p_peek)
            p[arm].append(p_min)
            p_int.append(p_min)
        
        # check if there is at-least one arm with p>0.05, then pull,
        # otherwise pull arm with highest ucb
        p_arm = [i for i in range(num_arms) if p_int[i]>0.05]
        print(arm_sort_ucb)
        print(p_int)
        if p_arm:
            # we only pull the arm if we know that the difference is
            # not statistically significant
            for arm in arm_sort_ucb:
                if p_int[arm]>0.05:
                    bandit.pull_arm(arm)
                    break
        else:
            print("pulled")
            bandit.pull_arm(arm_sort_ucb[0])
    
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = num_obs
    trt_dist_lis = trt_dist_list[:num_arms]
    ucb_bandit = Bandit(name='ucb_peek',
                        num_arms=num_arms,
                        trt_dist_list=trt_dist_lis)
    ucb_peek(bandit=ucb_bandit,
              num_rounds=num_rounds,
              num_arms=num_arms)