from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, thompson_arm_pull, bayesian_update
import numpy as np
from math import sqrt

    
def thompson_sampling(bandit, num_rounds):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    print("---------------Running Thompson Sampling ---------------")
    
    # allocate one subject to each arm (We can remove this later rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm, propensity=1/bandit.num_arms)
    
    # we initialize the distributions of arms to normal inverse chi squared
    # distibution. Each arm has params (m, k, v, s).
    # m is mean and s is variance
    
    num_arms = bandit.num_arms
    
    prior_params = [(bandit.avg_reward_tracker[i], 1,
                     bandit.var_est_tracker[i],
                     1) for i in
                    range(
    num_arms)]
    
    
    for rnd in range(num_rounds-(2*num_arms)):
        # we store all sampled values in this list
        arm_means = [i[0] for i in prior_params]
        arm_vars = [i[3] for i in prior_params]
        chosen_arm, prop_lis = thompson_arm_pull(m=arm_means, s=arm_vars,
                                                 type_of_pull='monte_carlo')
        # That chosen arm is pulled to observe reward
        bandit.pull_arm(chosen_arm, prop_lis[chosen_arm])
        reward = bandit.reward_tracker[-1]
        # posterior distribution is estimated using bayesian update
        prior_params[chosen_arm] = bayesian_update(prior_params[chosen_arm],
                                                   reward)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = 10
    trt_dist_lis_th = trt_dist_list[:num_arms]
    thompson_bandit = Bandit(name='thompson_sampling',
                            num_arms=num_arms,
                            trt_dist_list=trt_dist_lis_th)
    thompson_sampling(thompson_bandit, num_rounds=num_rounds)
