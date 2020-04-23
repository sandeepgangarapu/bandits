from bandits.ucb_inf.algorithms.ab_testing import ab_testing
from bandits.ucb_inf.algorithms.ucb_inf import ucb_inf
from bandits.ucb_inf.algorithms.thomp_inf_eps import thomp_inf_eps
from bandits.algorithms.thompson_sampling import thompson_sampling
from bandits.ucb_inf.algorithms.ucb_inf_eps import ucb_inf_eps
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.ucb_inf.stats import regret_outcome, mse_outcome, prop_mse
import pandas as pd
import numpy as np
from math import sqrt


def create_group_dict(group, outcome, alg, ite):
    dict = {"group": group, 'outcome': outcome,
            'alg': alg,
            'ite': ite}
    df = pd.DataFrame(dict)
    return df


def create_regret_dict(regret, mean_mse, var_mse, alg, ite):
    dict = {"regret": regret,
            'mean_mse': mean_mse,
            'var_mse': var_mse,
            'alg': alg,
            'ite': ite}
    df = pd.DataFrame(dict)
    return df


def create_prop_dict(mse, alg, ite, mean_est, group):
    dict = {'mse': mse,
            'alg': alg,
            'ite': ite,
            'mean_est': mean_est,
            'group': group}
    df = pd.DataFrame(dict)
    return df

def append_method_df(group, outcome, alg_name, ite, arm_means, arm_vars,
                     group_outcome_df, regret_mse_df):
    sub_grp_df = create_group_dict(group, outcome, alg_name, ite)
    regret = regret_outcome(group, outcome)
    mean_mse, var_mse = mse_outcome(group, outcome, arm_means, arm_vars)
    sub_regret_mse_df = create_regret_dict(regret, mean_mse, var_mse,
                                           alg_name, ite)
    group_outcome_df = group_outcome_df.append(sub_grp_df)
    regret_mse_df = regret_mse_df.append(sub_regret_mse_df)
    return group_outcome_df, regret_mse_df


def append_prop_df(group, mean_est, alg_name, ite, arm_means, prop_mse_df):
    mean_mse = prop_mse(group, mean_est, arm_means)
    sub_mse_df = create_prop_dict(mean_mse, alg_name, ite, mean_est, group)
    prop_mse_df = prop_mse_df.append(sub_mse_df)
    return prop_mse_df


def main():
    # Parameters
    num_ite = 20
    save_1 = True
    np.random.seed(seed=99)
    num_arms = 10
    num_subjects = 2000
    arm_means = np.random.uniform(0, 5, num_arms)
    arm_means[3] = 0.25
    arm_means[9] = 3
    arm_vars = np.random.uniform(0, 5, num_arms)
    arm_vars[8] = 1
    # arm_means = [2,4]
    # arm_vars=[2,1]
    print(arm_means)
    print(arm_vars)
    perc_ab_for_mixucb = 0.2
    outcome_lis_of_lis = []
    size = 100000
    
    # knobs
    ab_test_knob = False
    ucb_knob = False
    eps_greedy_knob = False
    ucb_inf_eps_knob = False
    ucb_inf_knob = False
    thomp_knob = True
    thomp_inf_eps_knob = True
    
    
    # Files to save
    group_outcome_df = pd.DataFrame()
    regret_mse_df = pd.DataFrame()
    prop_mse_df = pd.DataFrame()


    for ite in range(num_ite):
        print("------------------ ITERATION ", ite, "------------------")
        for arm in range(num_arms):
            outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                                       scale=sqrt(arm_vars[arm]),
                                                       size=size))
        
        if ab_test_knob:
            ab_group, ab_outcome = ab_testing(arm_means, arm_vars,
                                              num_subjects,
                                              post_allocation=True)
            group_outcome_df, regret_mse_df = append_method_df(ab_group,
                                                               ab_outcome,
                                                               "ab", ite,
                                                               arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)
        
        if ucb_knob:
            ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                                trt_dist_list=outcome_lis_of_lis)
            ucb_bandit = ucb_naive(bandit=ucb_bandit, num_rounds=num_subjects,
                                   num_arms=num_arms)
            ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(ucb_group,
                                                               ucb_outcome,
                                                               "ucb",
                                                               ite, arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)
        
        if eps_greedy_knob:
            eps_bandit = Bandit(name='eps_bandit', num_arms=num_arms,
                                trt_dist_list=outcome_lis_of_lis)
            eps_bandit = epsilon_greedy(epsilon=0.2, bandit=eps_bandit,
                                        num_rounds=num_subjects,
                                        num_arms=num_arms)
            eps_group, eps_outcome = eps_bandit.arm_tracker, eps_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(eps_group,
                                                               eps_outcome,
                                                               "eps",
                                                               ite, arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)
        
        if ucb_inf_knob:
            ucb_inf_bandit = Bandit(name='ucb_inf', num_arms=num_arms,
                                   trt_dist_list=outcome_lis_of_lis)
            ucb_inf_bandit = ucb_inf(ucb_inf_bandit, num_subjects, perc_ab=perc_ab_for_mixucb)
            ucb_inf_group, ucb_inf_outcome = ucb_inf_bandit.arm_tracker, ucb_inf_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(ucb_inf_group,
                                                               ucb_inf_outcome,
                                                               "ucb_inf",
                                                               ite, arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)
        
        if ucb_inf_eps_knob:
            ucb_inf_eps_bandit = Bandit(name='ucb_inf_eps', num_arms=num_arms,
                                        trt_dist_list=outcome_lis_of_lis)
    
            ucb_inf_eps_bandit = ucb_inf_eps(
                ucb_inf_eps_bandit,
                num_subjects,
                chi=1)
            ucb_inf_eps_group, ucb_inf_eps_outcome = ucb_inf_eps_bandit.arm_tracker, ucb_inf_eps_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(
                ucb_inf_eps_group,
                ucb_inf_eps_outcome, "ucb_inf_eps",  ite, arm_means, arm_vars,
                group_outcome_df, regret_mse_df)
        
        if thomp_knob:
            thomp_bandit = Bandit(name='thomp', num_arms=num_arms,
                                        trt_dist_list=outcome_lis_of_lis)
            thomp_bandit = thompson_sampling(thomp_bandit, num_subjects)
            thomp_group, thomp_outcome = thomp_bandit.arm_tracker, thomp_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(thomp_group,
                                                               thomp_outcome,
                                                               "thomp",
                                                               ite, arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)

            prop_mse_df = append_prop_df(thomp_group, thomp_bandit.ipw_tracker, "thomp_ipw", ite, arm_means,
                                         prop_mse_df)
            prop_mse_df = append_prop_df(thomp_group, thomp_bandit.aipw_tracker, "thomp_aipw", ite, arm_means,
                                         prop_mse_df)

        if thomp_inf_eps_knob:
            thomp_inf_eps_bandit = Bandit(name='thomp_inf_eps', num_arms=num_arms,
                                          trt_dist_list=outcome_lis_of_lis)
            thomp_inf_eps_bandit = thomp_inf_eps(thomp_inf_eps_bandit, num_subjects)
            thomp_inf_eps_group, thomp_inf_eps_outcome = thomp_inf_eps_bandit.arm_tracker, thomp_inf_eps_bandit.reward_tracker
            group_outcome_df, regret_mse_df = append_method_df(thomp_inf_eps_group,
                                                               thomp_inf_eps_outcome,
                                                               "thomp_inf_eps",
                                                               ite, arm_means,
                                                               arm_vars,
                                                               group_outcome_df,
                                                               regret_mse_df)
            prop_mse_df = append_prop_df(thomp_inf_eps_group, thomp_inf_eps_bandit.ipw_tracker, "thomp_inf_eps_ipw",
                                         ite, arm_means, prop_mse_df)
            prop_mse_df = append_prop_df(thomp_inf_eps_group, thomp_inf_eps_bandit.aipw_tracker, "thomp_inf_eps_aipw",
                                         ite, arm_means, prop_mse_df)

        if save_1:
            group_outcome_df.to_csv("Output/group_outcome_prop.csv",
                                    index=False)
            regret_mse_df.to_csv("Output/regret_prop_t.csv", index=False)
            prop_mse_df.to_csv("Output/ipw_aipw_prop.csv", index=False)
        else:
            group_outcome_df.to_csv("Output/group_outcome_sim.csv", index=False)
            regret_mse_df.to_csv("Output/regret_mse.csv", index=False)
        

if __name__ == '__main__':
    main()
