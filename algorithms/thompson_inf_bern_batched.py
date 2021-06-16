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


def thomp_inf_bern_batched(bandit, num_rounds, xi=0.05, cap_prop=False,
                           batch_size=100):
    """Function that reproduces the steps involved in Thompson sampling
        algorithm"""
    print("---------------Running Thompson Inf Bern Batched ---------------")

    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    num_initial_pulls = int(batch_size/num_arms)
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            bandit.pull_arm(arm)

    # beta bernoulli updation process
    prior_params = [[1, 1] for i in range(num_arms)]
    var_allocs = 0
    for batch in range(int(num_rounds/batch_size)-1):
        arm_counts = [0 for i in range(num_arms)]
        rewards = [0 for i in range(num_arms)]
        eps_n = calc_eps_n(bandit, xi)
        # variance based, we pick arm proportional to the variance of
        # mean estimate, i.e. s^2/n
        var_of_mean_est = \
            np.array(bandit.var_est_tracker) / np.array(
                bandit.arm_pull_tracker)
        # normalizing std.err square for calculating probabilities
        norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
        for j in range(batch_size):
            if np.random.uniform(0, 1) < eps_n:
                var_allocs += 1
                arm_var = np.random.choice(list(range(bandit.num_arms)),
                                           p=norm_prob)
                arm_counts[arm_var] += 1
                bandit.pull_arm(arm_var)
                x = bandit.reward_tracker[-1]
                rewards[arm_var] += x
            else:
                chosen_arm = thompson_arm_pull_bern(param_lis=prior_params)
                arm_counts[chosen_arm] += 1
                bandit.pull_arm(chosen_arm)
                x = bandit.reward_tracker[-1]
                rewards[chosen_arm] += x
        
        # We calculate the posterior parameters of the beta distribution
        # after batch allocation is done
        for arm in range(num_arms):
            prior_params[arm][0] += rewards[arm]
            prior_params[arm][1] += arm_counts[arm] - rewards[arm]
            
    return bandit

if __name__ == '__main__':
    # Define bandit
    for i in range(10):
        num_rounds = 3000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.1, 0.2, 0.3],
                                 dist_type='Bernoulli')
        thomp_inf_bern_batched(thompson_bandit, num_rounds=num_rounds, cap_prop=False)
