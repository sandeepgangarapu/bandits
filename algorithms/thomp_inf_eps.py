import numpy as np
from bandits.utils import thompson_arm_pull, bayesian_update


def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
        bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms * xi)
    eps_n = eta / (1 + eta)
    return eps_n


def thomp_inf_eps(bandit, num_rounds, xi=1):
    print("---------------Running Thompson Sampling INF EPS ---------------")

    # allocate one subject to each arm (We can remove this later rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm, propensity=1/bandit.num_arms)
    num_arms = bandit.num_arms

    prior_params = [(bandit.avg_reward_tracker[i], 1,
                     bandit.var_est_tracker[i], 1)
                    for i in range(num_arms)]
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_rounds-(2*num_arms)):
        # perc_ab of the time, we do variance based, otherwise UCB
        if np.random.uniform(0, 1) < calc_eps_n(bandit, xi):
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = np.array([i[3] for i in prior_params])/ \
                              np.array(
                    bandit.arm_pull_tracker)
            # normalizing std.err square for calculating probabilities
            norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
            arm_var = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
            bandit.pull_arm(arm_var, propensity=norm_prob[arm_var])
            reward = bandit.reward_tracker[-1]
            # posterior distribution is estimated using bayesian update
            prior_params[arm_var] = bayesian_update(
                prior_params[arm_var],
                reward)
        else:
            arm_means = [i[0] for i in prior_params]
            arm_vars = [i[3] for i in prior_params]
            chosen_arm, prop_lis = thompson_arm_pull(m=arm_means, s=arm_vars,
                                                     type_of_pull='monte_carlo')
            # That chosen arm is pulled to observe reward
            bandit.pull_arm(chosen_arm, prop_lis[chosen_arm])
            reward = bandit.reward_tracker[-1]
            # posterior distribution is estimated using bayesian update
            prior_params[chosen_arm] = bayesian_update(
                prior_params[chosen_arm],
                reward)
    return bandit