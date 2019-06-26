from bandits.MAB_VAR.ab_testing import ab_testing
from bandits.MAB_VAR.peek_ab import peeking_ab_testing
from bandits.MAB_VAR.vanilla_mix import vanilla_mixed_UCB
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.bandit import Bandit
from bandits.utils import create_distributions_vanilla, create_distributions_custom
from bandits.MAB_VAR.stats import overall_stats, regret_outcome, rmse_outcome
import pandas as pd
import numpy as np


if __name__ == '__main__':
    
    num_arms = 10
    arm_means = [0]
    other_means = np.random.uniform(0, 5, num_arms-1)
    arm_means.extend(other_means)
    print(np.argmax(arm_means))
    print(arm_means)
    arm_vars = np.random.uniform(0, 3, num_arms)
    outcome_lis_of_lis = create_distributions_custom(arm_means=arm_means, arm_vars=arm_vars)
    ab_group, ab_outcome = ab_testing(outcome_lis_of_lis, post_allocation=True)
    peek_group, peek_outcome = peeking_ab_testing(outcome_lis_of_lis,
                                                  post_allocation=True)
    mix_group, mix_outcome = vanilla_mixed_UCB(outcome_lis_of_lis)
    print("---------------Running UCB testing---------------")
    ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)
    ucb_bandit = ucb_naive(bandit=ucb_bandit,
                           num_rounds=len(outcome_lis_of_lis[0]),
                           num_arms=num_arms)
    ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
    # ab, peek, ucb, mix = map(overall_stats, [ab_outcome, peek_outcome,
    #                                          ucb_outcome, mix_outcome])
    print("---------------Calculating Regret---------------")
    ab_regret, peek_regret, ucb_regret, mix_regret = map(regret_outcome,
                             [ab_group, peek_group, ucb_group, mix_group],
                             [ab_outcome, peek_outcome, ucb_outcome, mix_outcome])
    print("---------------Calculating RMSE---------------")
    ab_mean_rmse, ab_var_rmse = rmse_outcome(ab_group, ab_outcome,
                                             arm_means, arm_vars)
    peek_mean_rmse, peek_var_rmse = rmse_outcome(peek_group, peek_outcome,
                                             arm_means, arm_vars)
    ucb_mean_rmse, ucb_var_rmse = rmse_outcome(ucb_group, ucb_outcome,
                                             arm_means, arm_vars)
    mix_mean_rmse, mix_var_rmse = rmse_outcome(mix_group, mix_outcome,
                                             arm_means, arm_vars)
    
    output = [ab_regret, peek_regret, ucb_regret, mix_regret, ab_mean_rmse, ab_var_rmse, peek_mean_rmse, peek_var_rmse, ucb_mean_rmse,\
    ucb_var_rmse, mix_mean_rmse, mix_var_rmse]
    
    df = pd.DataFrame(output).T
    df.columns = ["ab_regret", "peek_regret", "ucb_regret", "mix_regret", "ab_mean_rmse", "ab_var_rmse",
                  "peek_mean_rmse", "peek_var_rmse", "ucb_mean_rmse", "ucb_var_rmse", "mix_mean_rmse", "mix_var_rmse"]
    df.to_csv("output.csv", index=False)
    group_assignment = [ab_group, ucb_group, mix_group]
    df = pd.DataFrame(group_assignment).T
    df.columns = ["ab_group", "ucb_group", "mix_group"]
    df.to_csv("groups.csv", index=False)
    # ab = pd.DataFrame(ab)
    # peek = pd.DataFrame(peek)
    # ucb = pd.DataFrame(ucb)
    # mix = pd.DataFrame(mix)
    #
    # ab.to_csv("ab.csv", index=False)
    # peek.to_csv("peek.csv", index=False)
    # mix.to_csv("mix.csv", index=False)
    # ucb.to_csv("ucb.csv", index=False)

