from bandits.MAB_VAR.ab_testing import ab_testing
from bandits.MAB_VAR.vanilla_mix import vanilla_mixed_UCB
from bandits.MAB_VAR.var_mix_thompson import mixed_thompson
from bandits.MAB_VAR.var_mix_prop_thompson import mixed_prop_thompson
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.MAB_VAR.stats import regret_outcome, rmse_outcome
import pandas as pd
import numpy as np
from math import sqrt


if __name__ == '__main__':
    
    np.random.seed(seed=32353534)
    # knobs
    save_output = True
    num_arms = 10
    num_subjects = 2000
    arm_means = [0]
    other_means = np.random.uniform(0, 5, num_arms-1)
    arm_means.extend(other_means)
    print(np.argmax(arm_means))
    print(arm_means)
    arm_vars = np.random.uniform(0, 5, num_arms)
    print(arm_vars)
    perc_ab_for_mixucb = 0.2
    ab_group, ab_outcome = ab_testing(arm_means, arm_vars,
                                      num_subjects, post_allocation=True)
    mix_group, mix_outcome = mixed_thompson(arm_means, arm_vars,
                                               num_subjects,
                                               perc_ab=perc_ab_for_mixucb)

    mix_prop_group, mix_prop_outcome = mixed_prop_thompson(arm_means, arm_vars,
                                            num_subjects,
                                            perc_ab=perc_ab_for_mixucb)
    
    print("---------------Running UCB Bandit---------------")
    outcome_lis_of_lis = []
    size = 10000
    for arm in range(num_arms):
        outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                        scale=sqrt(arm_vars[arm]), size=size))
            
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
    print("---------------Calculating Regret---------------")
    ab_regret, ucb_regret, eps_regret, mix_regret, mix_prop_regret = map(
        regret_outcome,
        [ab_group, ucb_group, eps_group, mix_group, mix_prop_group],
        [ab_outcome, ucb_outcome, eps_outcome,  mix_outcome, mix_prop_outcome])
    print("---------------Calculating RMSE---------------")
    ab_mean_rmse, ab_var_rmse = rmse_outcome(ab_group, ab_outcome,
                                             arm_means, arm_vars)
    ucb_mean_rmse, ucb_var_rmse = rmse_outcome(ucb_group, ucb_outcome,
                                             arm_means, arm_vars)
    eps_mean_rmse, eps_var_rmse = rmse_outcome(eps_group, eps_outcome,
                                               arm_means, arm_vars)
    mix_mean_rmse, mix_var_rmse = rmse_outcome(mix_group, mix_outcome,
                                             arm_means, arm_vars)
    mix_prop_mean_rmse, mix_prop_var_rmse = rmse_outcome(mix_prop_group,
                                                         mix_prop_outcome,
                                                         arm_means, arm_vars)
    
    
    output = [ab_regret, ucb_regret, eps_regret, mix_regret, mix_prop_regret,
              ab_mean_rmse, ab_var_rmse, ucb_mean_rmse, ucb_var_rmse,
              eps_mean_rmse, eps_var_rmse, mix_mean_rmse, mix_var_rmse,
              mix_prop_mean_rmse, mix_prop_var_rmse]
    
    df1 = pd.DataFrame(output).T
    df1.columns = ["ab_regret", "ucb_regret", "eps_regret", "mix_regret",
                   "mix_prop_regret", "ab_mean_rmse",
                   "ab_var_rmse", "ucb_mean_rmse", "ucb_var_rmse", "eps_mean_rmse",
                   "eps_var_rmse", "mix_mean_rmse", "mix_var_rmse", "mix_prop_mean_rmse", "mix_prop_var_rmse"]
    
    group_assignment = [ab_group, ucb_group, eps_group, mix_group,
                        mix_prop_group]
    df2 = pd.DataFrame(group_assignment).T
    df2.columns = ["ab_group", "ucb_group", "eps_group", "mix_group",
                   "mix_prop_group"]
    if save_output:
        df1.to_csv("output.csv", index=False)
        df2.to_csv("groups.csv", index=False)


