import numpy as np
from utils import thompson_arm_pull_bern, capped_prop
from bandit import Bandit
from math import sqrt


def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
        bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms * xi)
    eps_n = eta / (1 + eta)
    return eps_n


def thomp_inf_bern_batched(bandit, num_rounds, xi=0.05, cap_prop=False,
                           batch_size=100, type_of_pull='single'):
    """Function that reproduces the steps involved in Thompson sampling
        algorithm"""
    print("---------------Running Thompson Inf Bern Batched ---------------")

    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    # beta bernoulli updation process
    prior_params = [[1, 1] for i in range(num_arms)]
    num_initial_pulls = int(batch_size/num_arms)
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)
            x = bandit.reward_tracker[-1]
            # We calculate the posterior parameters of the beta distribution
            prior_params[arm][0] += x
            prior_params[arm][1] += 1-x
    var_allocs = 0
    for batch in range(int(num_rounds/batch_size)-1):
        arm_counts = [0 for i in range(num_arms)]
        rewards = [0 for i in range(num_arms)]
        # cap will be same for the entire batch
        cap = 0.1 / sqrt(((batch+1)*batch_size) + 1)
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
                if cap_prop:
                    norm_prob = capped_prop(norm_prob, cap)
                    arm_var = np.random.choice(range(bandit.num_arms),
                                               p=norm_prob)
                bandit.pull_arm(arm_var, prop_lis=norm_prob)
                x = bandit.reward_tracker[-1]
                rewards[arm_var] += x
            else:
                if cap_prop:
                    chosen_arm, prop_lis = thompson_arm_pull_bern(
                        param_lis=prior_params, type_of_pull=type_of_pull, cap_prop=cap)
                else:
                    chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
            
                bandit.pull_arm(chosen_arm, prop_lis)
                arm_counts[chosen_arm] += 1
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
    for i in range(100):
        num_rounds = 2000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.37098621, 0.33080171, 0.1699615, 0.18902466, 0.6743146],
                                 dist_type='Bernoulli')
        thomp_inf_bern_batched(thompson_bandit,
                                       num_rounds=num_rounds, cap_prop=True,
                                       type_of_pull='monte_carlo')
