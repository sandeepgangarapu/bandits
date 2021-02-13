import numpy as np
from utils import thompson_arm_pull_bern
from bandit import Bandit
from math import sqrt


def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
        bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms * xi)
    eps_n = eta / (1 + eta)
    return eps_n


def thomp_inf_bern(bandit, num_rounds, xi=0.05, type_of_pull='single', cap_prop=False):
    """Function that reproduces the steps involved in Thompson sampling
        algorithm"""
    print("---------------Running Thompson Inf Bern ---------------")

    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    num_initial_pulls = 3
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)

    # beta bernoulli updation process
    prior_params = [[1, 1] for i in range(num_arms)]
    var_allocs = 0
    for rnd in range(int(num_rounds - (num_initial_pulls * num_arms))):
        if np.random.uniform(0, 1) < calc_eps_n(bandit, xi):
            var_allocs += 1
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = \
                np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)
            # normalizing std.err square for calculating probabilities
            norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
            arm_var = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
            if type_of_pull == 'monte_carlo':
                # this is to cap probility of allocation as per Athey's algortihm
                if cap_prop:
                    # the cap is defined in the paper
                    cap = 0.1 / sqrt(rnd + 1)
                    if np.min(norm_prob) < cap:
                        # every arm gets cap propensity at the start, the remaining propensity is shared proportional to the
                        # true propensity
                        remaining_prop = 1 - cap * bandit.num_arms
                        norm_prob = cap + np.array(norm_prob) * remaining_prop
                        arm_var = np.random.choice(range(bandit.num_arms),
                                                   p=norm_prob)
                bandit.pull_arm(arm_var, prop_lis=norm_prob)
            else:
                bandit.pull_arm(arm_var)
            x = bandit.reward_tracker[-1]
            # We calculate the posterior parameters of the beta distribution
            prior_params[arm_var][0] += x
            prior_params[arm_var][1] += 1 - x
        else:
            if type_of_pull == 'monte_carlo':
                chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
                # this is to cap probility of allocation as per Athey's algortihm
                if cap_prop:
                    cap = 0.1 / sqrt(rnd + 1)
                    chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull,
                                                                  cap_prop=cap)
                bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
            else:
                chosen_arm = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
                bandit.pull_arm(chosen_arm)
            x = bandit.reward_tracker[-1]
            # We calculate the posterior parameters of the beta distribution
            prior_params[chosen_arm][0] += x
            prior_params[chosen_arm][1] += 1 - x
    return bandit

if __name__ == '__main__':
    # Define bandit
    for i in range(10):
        num_rounds = 3000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.1, 0.2, 0.3],
                                 dist_type='Bernoulli')
        thomp_inf_bern(thompson_bandit, num_rounds=num_rounds, type_of_pull="monte_carlo", cap_prop=False)
