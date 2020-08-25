import numpy as np
from bandits.utils import ucb_value_naive


def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms*xi)
    eps_n = eta/(1+eta)
    return eps_n


def ucb_inf_eps(bandit, num_rounds, xi=1, type_of_pull='single'):
    print("---------------Running UCB INF EPS ---------------")
    
    # allocate one subject to each arm (UCB rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_rounds - (2 * bandit.num_arms)):
        # perc_ab of the time, we do variance based, otherwise UCB
        if np.random.uniform(0, 1) < calc_eps_n(bandit, xi):
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
                bandit.pull_arm(arm_var, prop_lis=norm_prob)
            else:
                bandit.pull_arm(arm_var)
        else:
            if type_of_pull == 'monte_carlo':
                arm_max_ucb, prop_lis = ucb_value_naive(bandit.num_arms, num_rounds,
                                                        bandit.arm_pull_tracker,
                                                        bandit.avg_reward_tracker,
                                                        bandit.var_est_tracker,
                                                        type_of_pull='monte_carlo')
                bandit.pull_arm(arm_max_ucb, prop_lis)
            else:
                arm_max_ucb = ucb_value_naive(bandit.num_arms, num_rounds,
                                              bandit.arm_pull_tracker,
                                              bandit.avg_reward_tracker,
                                              bandit.var_est_tracker)
                bandit.pull_arm(arm_max_ucb)

    return bandit
