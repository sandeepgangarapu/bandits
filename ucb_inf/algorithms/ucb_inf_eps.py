import numpy as np
from bandits.utils import ucb_value_naive

def calc_eps_n(bandit, chi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms*chi)
    eps_n = eta/(1+eta)
    print(eps_n)
    return eps_n


def ucb_inf_eps(bandit, num_subjects, chi=1):
    print("---------------Running UCB INF EPS ---------------")
    
    # allocate one subject to each arm (UCB rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects - (2 * bandit.num_arms)):
        # perc_ab of the time, we do variance based, otherwise UCB
        if np.random.uniform(0, 1) < calc_eps_n(bandit, chi):
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = \
                np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)
            # normalizing std.err square for calculating probabilities
            norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
            arm_var = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
            bandit.pull_arm(arm_var)
        else:
            ucb = ucb_value_naive(num_arms=bandit.num_arms,
                                  num_rounds=num_subjects,
                                  arm_pull_tracker=bandit.arm_pull_tracker,
                                  avg_reward_tracker=bandit.avg_reward_tracker)
            arm_max_ucb = np.argmax(ucb)
            bandit.pull_arm(arm_max_ucb)
    
    return bandit.arm_tracker, bandit.reward_tracker