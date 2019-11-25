from bandits.MAB_VAR.algorithms.ab_testing import ab_testing
from bandits.MAB_VAR.algorithms.ucb_inf_trt_max import ucb_inf_trt_max_alg
from bandits.MAB_VAR.algorithms.ucb_inf_trt_prop import ucb_inf_trt_prop_alg
from bandits.MAB_VAR.algorithms.ucb_inf_var_prop import ucb_inf_var_prop_alg
from bandits.MAB_VAR.algorithms.ucb_inf_var_max import ucb_inf_var_max_alg
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.MAB_VAR.stats import regret_outcome, mse_outcome, var_stats
import pandas as pd
import numpy as np
from math import sqrt


if __name__ == '__main__':

    # seeds 23423
    np.random.seed(seed=636346346)
    # knobs
    save_output = True
    num_arms = 10
    num_subjects = 2000
    arm_means = [0]
    other_means = np.random.uniform(0, 5, num_arms-1)
    arm_means.extend(other_means)
    arm_means[3] = 0.25
    print(np.argmax(arm_means))
    arm_vars = np.random.uniform(0, 5, num_arms)
    print(arm_means)
    print(arm_vars)
    perc_ab_for_mixucb = 0.2
    outcome_lis_of_lis = []
    size = 100000
    ab_test = True
    ucb = True
    eps_greedy = True
    ucb_inf_var_max = True
    ucb_inf_var_prop = True
    ucb_inf_trt_max = True
    ucb_inf_trt_prop = True
    
    for arm in range(num_arms):
        outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                                   scale=sqrt(arm_vars[arm]),
                                                   size=size))
        
    if ab_testing:
        ab_group, ab_outcome = ab_testing(arm_means, arm_vars,
                                          num_subjects, post_allocation=True)
    
    if ucb:
        ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                            trt_dist_list=outcome_lis_of_lis)
        ucb_bandit = ucb_naive(bandit=ucb_bandit, num_rounds=num_subjects,
                               num_arms=num_arms)
        ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
    
   
    if eps_greedy:
        eps_bandit = Bandit(name='eps_bandit', num_arms=num_arms,
                            trt_dist_list=outcome_lis_of_lis)
        eps_bandit = epsilon_greedy(epsilon=0.2, bandit=eps_bandit,
                                    num_rounds=num_subjects, num_arms=num_arms)
        eps_group, eps_outcome = eps_bandit.arm_tracker, eps_bandit.reward_tracker

    if ucb_inf_var_max:
        ucb_inf_var_max_bandit = Bandit(name='ucb_inf_var_max',
                                        num_arms=num_arms,
                                        trt_dist_list=outcome_lis_of_lis)
    
        ucb_inf_var_max_group, ucb_inf_var_max_outcome = \
            ucb_inf_var_max_alg(
                ucb_inf_var_max_bandit, num_subjects,
                perc_ab=perc_ab_for_mixucb)
        
    if ucb_inf_var_prop:
        ucb_inf_var_prop_bandit = Bandit(name='ucb_inf_var_prop',
                                  num_arms=num_arms,
                                  trt_dist_list=outcome_lis_of_lis)
    
        ucb_inf_var_prop_group, ucb_inf_var_prop_outcome = \
            ucb_inf_var_prop_alg(
            ucb_inf_var_prop_bandit, num_subjects,
            perc_ab=perc_ab_for_mixucb)
        print("ucb_inf_var_prop_bandit.avg_reward_tracker")
        print(ucb_inf_var_prop_bandit.avg_reward_tracker)
    if ucb_inf_trt_max:
        ucb_inf_trt_max_bandit = Bandit(name='ucb_inf_trt_max',
                                        num_arms=num_arms,
                                        trt_dist_list=outcome_lis_of_lis)
    
        ucb_inf_trt_max_group, ucb_inf_trt_max_outcome = \
            ucb_inf_trt_max_alg(
                ucb_inf_trt_max_bandit, num_subjects,
                perc_ab=perc_ab_for_mixucb)
        
        
    if ucb_inf_trt_prop:
        ucb_inf_trt_prop_bandit = Bandit(name='ucb_inf_trt_prop',
                                         num_arms=num_arms,
                                         trt_dist_list=outcome_lis_of_lis)
    
        ucb_inf_trt_prop_group, ucb_inf_trt_prop_outcome = \
            ucb_inf_trt_prop_alg(
                ucb_inf_trt_prop_bandit, num_subjects,
                perc_ab=perc_ab_for_mixucb)
    
    
            
    print("---------------Calculating Regret---------------")
    
    
    ab_regret, ucb_regret, eps_regret, ucb_inf_var_max_regret, \
    ucb_inf_var_prop_regret, ucb_inf_trt_max_regret, ucb_inf_trt_prop_regret= \
        map(
        regret_outcome,
        [ab_group, ucb_group, eps_group, ucb_inf_var_max_group,
         ucb_inf_var_prop_group, ucb_inf_trt_max_group, ucb_inf_trt_prop_group],
        [ab_outcome, ucb_outcome, eps_outcome,  ucb_inf_var_max_outcome,
         ucb_inf_var_prop_outcome, ucb_inf_trt_max_outcome,
         ucb_inf_trt_prop_outcome])
    
    
    print("---------------Calculating MSE---------------")
    ab_mean_mse, ab_var_mse = mse_outcome(ab_group, ab_outcome,
                                             arm_means, arm_vars)
    ucb_mean_mse, ucb_var_mse = mse_outcome(ucb_group, ucb_outcome,
                                             arm_means, arm_vars)
    eps_mean_mse, eps_var_mse = mse_outcome(eps_group, eps_outcome,
                                               arm_means, arm_vars)
    ucb_inf_var_max_mean_mse, ucb_inf_var_max_var_mse = mse_outcome(
        ucb_inf_var_max_group, ucb_inf_var_max_outcome, arm_means, arm_vars)

    ucb_inf_var_prop_mean_mse, ucb_inf_var_prop_var_mse = mse_outcome(
        ucb_inf_var_prop_group, ucb_inf_var_prop_outcome, arm_means, arm_vars)

    ucb_inf_trt_prop_mean_mse, ucb_inf_trt_prop_var_mse = mse_outcome(
        ucb_inf_trt_prop_group, ucb_inf_trt_prop_outcome, arm_means, arm_vars)

    ucb_inf_trt_max_mean_mse, ucb_inf_trt_max_var_mse = mse_outcome(
        ucb_inf_trt_max_group, ucb_inf_trt_max_outcome, arm_means, arm_vars)
    
    
    output = [ab_regret, ucb_regret, eps_regret, ucb_inf_var_max_regret,
              ucb_inf_var_prop_regret, ucb_inf_trt_max_regret, ucb_inf_trt_prop_regret,
              ab_mean_mse, ab_var_mse,
              ucb_mean_mse, ucb_var_mse,
              eps_mean_mse, eps_var_mse,
              ucb_inf_var_max_mean_mse, ucb_inf_var_max_var_mse,
              ucb_inf_var_prop_mean_mse, ucb_inf_var_prop_var_mse,
              ucb_inf_trt_prop_mean_mse, ucb_inf_trt_prop_var_mse,
              ucb_inf_trt_max_mean_mse, ucb_inf_trt_max_var_mse]
    
    df1 = pd.DataFrame(output).T
    df1.columns = ["ab_regret", "ucb_regret", "eps_regret", "ucb_inf_var_max_regret",
              "ucb_inf_var_prop_regret", "ucb_inf_trt_max_regret", "ucb_inf_trt_prop_regret",
              "ab_mean_mse", "ab_var_mse",
              "ucb_mean_mse", "ucb_var_mse",
              "eps_mean_mse", "eps_var_mse",
              "ucb_inf_var_max_mean_mse", "ucb_inf_var_max_var_mse",
              "ucb_inf_var_prop_mean_mse", "ucb_inf_var_prop_var_mse",
              "ucb_inf_trt_prop_mean_mse", "ucb_inf_trt_prop_var_mse",
              "ucb_inf_trt_max_mean_mse", "ucb_inf_trt_max_var_mse"]
    
    group_assignment = [ab_group, ucb_group, eps_group, ucb_inf_var_max_group,
         ucb_inf_var_prop_group, ucb_inf_trt_max_group, ucb_inf_trt_prop_group]
    df2 = pd.DataFrame(group_assignment).T
    df2.columns = ["ab_group", "ucb_group", "eps_group", "ucb_inf_var_max_group",
         "ucb_inf_var_prop_group", "ucb_inf_trt_max_group", "ucb_inf_trt_prop_group"]

    if save_output:
        df1.to_csv("output.csv", index=False)
        df2.to_csv("groups.csv", index=False)


