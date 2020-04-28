import numpy as np
from bandits.utils import ucb_value_naive


def ucb_inf_var_prop_alg(bandit, num_subjects, perc_ab=0.2):
    print("---------------Running UCB INF with prop var alloc---------------")
    
    # allocate one subject to each arm (UCB rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects - (2 * bandit.num_arms)):
        # perc_ab of the time, we do AB testing otherwise bandits
        if np.random.uniform(0, 1) < perc_ab:
            # now we pick the arm that has highest variance change
            # only if var change is higher than threshold, else we do UCB
            if np.max(bandit.var_change_tracker) < 0.01:
                ucb = ucb_value_naive(num_arms=bandit.num_arms,
                                      num_rounds=num_subjects,
                                      arm_pull_tracker=bandit.arm_pull_tracker,
                                      avg_reward_tracker=bandit.avg_reward_tracker)
                arm_max_ucb = np.argmax(ucb)
                bandit.pull_arm(arm_max_ucb)
            else:
                # now we pick the arm proportional to variance change of arm
                
                norm_prob = [v / sum(bandit.var_of_var_est_tracker)
                             for v in
                             bandit.var_of_var_est_tracker]
                arm = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
                bandit.pull_arm(arm)
        else:
            ucb = ucb_value_naive(num_arms=bandit.num_arms,
                                  num_rounds=num_subjects,
                                  arm_pull_tracker=bandit.arm_pull_tracker,
                                  avg_reward_tracker=bandit.avg_reward_tracker)
            arm_max_ucb = np.argmax(ucb)
            bandit.pull_arm(arm_max_ucb)
    
    return bandit.arm_tracker, bandit.reward_tracker