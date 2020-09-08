from bandit import Bandit
from utils import trt_dist_list, thompson_arm_pull, bayesian_update_normal_inv_gamma
import numpy as np
from math import sqrt

    
def thompson_sampling(bandit, num_rounds, type_of_pull='single'):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    print("---------------Running Thompson Sampling ---------------")
    
    # allocate one subject to each arm (We can remove this later rules)
    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    for ite in range(2):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)

    # we initialize the distributions of arms to normal inverse chi squared
    # distibution. Each arm has params (m, k, v, s).
    # m is mean and s is variance
    # changed the distribution to normal inverse gamma
    # https://www.coursera.org/lecture/bayesian/the-normal-gamma-conjugate-family-ncApT

    # prior = NormalGamma(m0, n0, s0^2, d0)
    # m0 = mean, n0=num_values, s0^2=sample variance, d0=n0-1(deg of freedom)
    prior_params = [(bandit.avg_reward_tracker[i], 2, bandit.var_est_tracker[i], 1) for i in
                    range(num_arms)]

    for rnd in range(int((num_rounds-(2*num_arms))/2)):
        # we divided it by 2 because we pull each arm twice to have a sample variance
        # we store all sampled values in this list
        arm_means = [i[0] for i in prior_params]
        arm_vars = [i[2] for i in prior_params]
        if type_of_pull == 'monte_carlo':
            chosen_arm, prop_lis = thompson_arm_pull(m=arm_means, s=arm_vars,
                                                 type_of_pull=type_of_pull)
        else:
            chosen_arm = thompson_arm_pull(m=arm_means, s=arm_vars,
                                                     type_of_pull=type_of_pull)
        # That chosen arm is pulled to observe reward
        # we pull the arm twice
        if type_of_pull == 'monte_carlo':
            bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
            bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
        else:
            bandit.pull_arm(chosen_arm)
            bandit.pull_arm(chosen_arm)

        reward_1 = bandit.reward_tracker[-1]
        reward_2 = bandit.reward_tracker[-2]
        # posterior distribution is estimated using bayesian update
        prior_params[chosen_arm] = bayesian_update_normal_inv_gamma(prior_params[chosen_arm],
                                                                    lis_x=[reward_1, reward_2])
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 1
    num_rounds = 50
    trt_dist_lis_th = trt_dist_list[:num_arms]
    thompson_bandit = Bandit(name='thompson_sampling',
                             num_arms=num_arms,
                             trt_dist_list=trt_dist_lis_th)
    thompson_sampling(thompson_bandit, num_rounds=num_rounds)
