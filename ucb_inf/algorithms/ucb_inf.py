import numpy as np
from bandits.utils import ucb_value_naive


def ucb_inf(bandit, num_subjects, perc_ab=0.2):
    print("---------------Running UCB INF---------------")
    
    # allocate one subject to each arm (UCB rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects - (2 * bandit.num_arms)):
        # perc_ab of the time, we do variance based, otherwise UCB
        if np.random.uniform(0, 1) < perc_ab:
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = \
                np.array(bandit.var_est_tracker)/np.array(
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
    
    return bandit