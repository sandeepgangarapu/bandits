from bandits.MAB_VAR.algorithms.ab_testing import ab_testing
from bandits.MAB_VAR.algorithms.trt_prop_variance import trt_prop_variance_est
from bandits.MAB_VAR.algorithms.var_mix_prop_thompson import mixed_prop_thompson
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.MAB_VAR.stats import regret_outcome, rmse_outcome, var_stats
import pandas as pd
import numpy as np
from math import sqrt


if __name__ == '__main__':

    # seeds 3523
    np.random.seed(seed=23423)
    # knobs
    save_output = True
    num_arms = 10
    num_subjects = 2000
    arm_means = [0]
    other_means = np.random.uniform(0, 5, num_arms-1)
    arm_means.extend(other_means)
    arm_means[3] = 0.35
    print(np.argmax(arm_means))
    print(arm_means)
    arm_vars = np.random.uniform(0, 5, num_arms)
    print(arm_vars)
    perc_ab_for_mixucb = 0.2
    outcome_lis_of_lis = []
    size = 100000
    for arm in range(num_arms):
        outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                                   scale=sqrt(arm_vars[arm]),
                                                   size=size))
        
        
    ab_group, ab_outcome = ab_testing(arm_means, arm_vars,
                                      num_subjects, post_allocation=True)

    trt_prop_var_bandit = Bandit(name='trt_prop_var_bandit', num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)
    
    trt_mix_prop_group, trt_mix_prop_outcome = trt_prop_variance_est(
        trt_prop_var_bandit, num_subjects, perc_ab=perc_ab_for_mixucb)

    mix_bandit = Bandit(name='mix_bandit', num_arms=num_arms,
                                 trt_dist_list=outcome_lis_of_lis)
    
    mix_prop_group, mix_prop_outcome = mixed_prop_thompson(mix_bandit,
                                            num_subjects,
                                            perc_ab=perc_ab_for_mixucb)
    
    print("---------------Running UCB Bandit---------------")
    
            
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
    ab_regret, ucb_regret, eps_regret, trt_prop_mix_regret, mix_prop_regret = \
        map(
        regret_outcome,
        [ab_group, ucb_group, eps_group, trt_mix_prop_group, mix_prop_group],
        [ab_outcome, ucb_outcome, eps_outcome,  trt_mix_prop_outcome, mix_prop_outcome])
    print("---------------Calculating RMSE---------------")
    ab_mean_rmse, ab_var_rmse = rmse_outcome(ab_group, ab_outcome,
                                             arm_means, arm_vars)
    ucb_mean_rmse, ucb_var_rmse = rmse_outcome(ucb_group, ucb_outcome,
                                             arm_means, arm_vars)
    eps_mean_rmse, eps_var_rmse = rmse_outcome(eps_group, eps_outcome,
                                               arm_means, arm_vars)
    trt_prop_mix_mean_rmse, trt_prop_mix_var_rmse = rmse_outcome(trt_mix_prop_group,
                                                        trt_mix_prop_outcome,
                                             arm_means, arm_vars)
    mix_prop_mean_rmse, mix_prop_var_rmse = rmse_outcome(mix_prop_group,
                                                         mix_prop_outcome,
                                                         arm_means, arm_vars)
    
    
    output = [ab_regret, ucb_regret, eps_regret, trt_prop_mix_regret, mix_prop_regret,
              ab_mean_rmse, ab_var_rmse, ucb_mean_rmse, ucb_var_rmse,
              eps_mean_rmse, eps_var_rmse, trt_prop_mix_mean_rmse, trt_prop_mix_var_rmse,
              mix_prop_mean_rmse, mix_prop_var_rmse]
    
    df1 = pd.DataFrame(output).T
    df1.columns = ["ab_regret", "ucb_regret", "eps_regret", "trt_prop_mix_regret",
                   "mix_prop_regret", "ab_mean_rmse",
                   "ab_var_rmse", "ucb_mean_rmse", "ucb_var_rmse", "eps_mean_rmse",
                   "eps_var_rmse", "trt_prop_mix_mean_rmse", "trt_prop_mix_var_rmse", "mix_prop_mean_rmse", "mix_prop_var_rmse"]
    
    group_assignment = [ab_group, ucb_group, eps_group, trt_mix_prop_group,
                        mix_prop_group]
    df2 = pd.DataFrame(group_assignment).T
    df2.columns = ["ab_group", "ucb_group", "eps_group", "trt_mix_prop_group",
                   "mix_prop_group"]

    group_assignment = [ab_group, ucb_group, eps_group, trt_mix_prop_group,
                        mix_prop_group]
    df2 = pd.DataFrame(group_assignment).T
    df2.columns = ["ab_group", "ucb_group", "eps_group", "trt_mix_prop_group",
                   "mix_prop_group"]

    # VARiance tracking
    ab_var, ucb_var, eps_var, mix_var, mix_prop_var = map(
        var_stats,
        [ab_group, ucb_group, eps_group, trt_mix_prop_group, mix_prop_group],
        [ab_outcome, ucb_outcome, eps_outcome, trt_mix_prop_outcome, mix_prop_outcome])


    arm = np.tile([i for i in range(num_arms)], len(ab_group))

    var_df = [arm, ab_var, ucb_var, eps_var, mix_var, mix_prop_var]
    df3 = pd.DataFrame(var_df).T
    df3.columns = ["arm", "ab_var", "ucb_var", "eps_var", "mix_var", "mix_prop_var"]

    if save_output:
        df1.to_csv("output.csv", index=False)
        df2.to_csv("groups.csv", index=False)
        df3.to_csv("var.csv", index=False)


