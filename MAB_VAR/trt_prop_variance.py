import numpy as np
from bandits.utils import treatment_outcome_grouping
from bandits.utils import ucb_value_naive
from math import sqrt



def trt_prop_variance_est(bandit, num_subjects, perc_ab=0.2):

    # allocate one subject to each arm (UCB rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)


    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects - (2 * bandit.num_arms)):
        # perc_ab of the time, we do AB testing otherwise bandits
        if np.random.uniform(0, 1) < perc_ab:
            # now we pick the arm proportional to variance of variance estimate
            
            norm_prob = [v / sum(bandit.trt_effect_var_of_var_est_tracker) for v in
                         bandit.trt_effect_var_of_var_est_tracker]            
            arm = np.random.choice(list(range(1, bandit.num_arms)), p=norm_prob)
            
            # Then we decide whether we pull control or the treatment arm
            # we do that by proportional to variance estimate
            
            probability_of_trt = \
                bandit.trt_effect_var_of_var_est_tracker[
                    arm]/bandit.trt_effect_var_of_var_est_tracker[
                    arm] + bandit.trt_effect_var_of_var_est_tracker[
                    0]
            
            arm = np.random.choice([arm, 0], p=[probability_of_trt,
                                                1-probability_of_trt])
            
            bandit.pull_arm(arm)
        
        else:
            
            ucb = ucb_value_naive(num_arms=bandit.num_arms,
                                  num_rounds=num_subjects,
                                  arm_pull_tracker=bandit.arm_pull_tracker,
                                  avg_reward_tracker=bandit.avg_reward_tracker)
            arm_max_ucb = np.argmax(ucb)
            bandit.pull_arm(arm_max_ucb)
    
    return bandit.arm_tracker, bandit.reward_tracker


