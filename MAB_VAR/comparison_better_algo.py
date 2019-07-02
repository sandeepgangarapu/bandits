from bandits.MAB_VAR.ab_testing import ab_testing
from bandits.MAB_VAR.vanilla_mix import vanilla_mixed_UCB
from bandits.MAB_VAR.var_mix_thompson import mixed_thompson
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.MAB_VAR.stats import regret_outcome, rmse_outcome
import pandas as pd
import numpy as np
from math import sqrt


def best_arm(group):
    arms = np.unique(group)
    grp_counts = [group.count(arm) for arm in range(len(arms))]
    b_arm = np.argmax(grp_counts)
    return b_arm
    
if __name__ == '__main__':
    num_iters = 100
    # knobs
    df = pd.DataFrame()
    for j in range(num_iters):
        print("--------------------------------------")
        print("Iteration no is ", j)
        print("--------------------------------------")
        save_output = True
        num_arms = 10
        num_subjects = 2000
        arm_means = [0]
        other_means = np.random.uniform(0, 5, num_arms - 1)
        arm_means.extend(other_means)
        arm_vars = np.random.uniform(0, 5, num_arms)
        b_arm = np.argmax(arm_means)
        perc_ab_for_mixucb = 0.2
        ab_group, ab_outcome = ab_testing(arm_means, arm_vars,
                                          num_subjects, post_allocation=True)
        mix_group, mix_outcome = mixed_thompson(arm_means, arm_vars,
                                                num_subjects,
                                                perc_ab=perc_ab_for_mixucb)
        print("---------------Running UCB Bandit---------------")
        outcome_lis_of_lis = []
        size = 10000
        for arm in range(num_arms):
            outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                                       scale=sqrt(
                                                           arm_vars[arm]),
                                                       size=size))
        ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                            trt_dist_list=outcome_lis_of_lis)
        ucb_bandit = ucb_naive(bandit=ucb_bandit, num_rounds=num_subjects,
                               num_arms=num_arms)
        ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
        print("---------------Running EPS Bandit---------------")
        
        eps_bandit = Bandit(name='eps_bandit', num_arms=num_arms,
                            trt_dist_list=outcome_lis_of_lis)
        eps_bandit = epsilon_greedy(epsilon=0.2, bandit=eps_bandit,
                                    num_rounds=num_subjects, num_arms=num_arms)
        eps_group, eps_outcome = eps_bandit.arm_tracker, eps_bandit.reward_tracker

        ab_barm, mix_barm, eps_barm, ucb_barm = map(best_arm, [ab_group,
                                                               mix_group,
                                                               eps_group, ucb_group])
        
        ab_var_barm, mix_var_barm, eps_var_barm, ucb_var_barm = arm_vars[
                                                                    ab_barm], arm_vars[mix_barm], arm_vars[eps_barm], arm_vars[ucb_barm]

        output = [b_arm, ab_barm, mix_barm, eps_barm, ucb_barm,
                     ab_var_barm, mix_var_barm, eps_var_barm, ucb_var_barm]
        df1 = pd.DataFrame(output).T
        df1.columns = ["b_arm", "ab_barm", "mix_barm", "eps_barm", "ucb_barm",
                       "ab_var_barm", "mix_var_barm", "eps_var_barm", "ucb_var_barm"]
        df = df.append(df1, ignore_index = True)
        df.to_csv("compare_output.csv", index=False)