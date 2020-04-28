from bandits.ucb_inf.algorithms.ucb_inf_eps import ucb_inf_eps
from bandits.bandit import Bandit
from bandits.ucb_inf.stats import regret_outcome, mse_outcome
import pandas as pd
import numpy as np
from math import sqrt


def create_group_dict(group, outcome, alg, ite, chi):
    dict = {"group": group, 'outcome': outcome,
            'alg': alg,
            'ite': ite,
            'chi': chi}
    df = pd.DataFrame(dict)
    return df


def create_regret_dict(regret, mean_mse, var_mse, alg, ite, chi):
    dict = {"regret": regret,
            'mean_mse': mean_mse,
            'var_mse': var_mse,
            'alg': alg,
            'ite': ite,
            'chi': chi}
    df = pd.DataFrame(dict)
    return df


def main():
    # Parameters
    num_ite = 1
    np.random.seed(seed=636346346)
    num_arms = 10
    num_subjects = 8000
    arm_means = np.random.uniform(0, 5, num_arms)
    arm_means[3] = 0.25
    arm_means[9] = 3
    arm_vars = np.random.uniform(0, 5, num_arms)
    arm_vars[8] = 1
    # arm_means = [2,4]
    # arm_vars=[2,1]
    print(arm_means)
    print(arm_vars)
    outcome_lis_of_lis = []
    size = 100000

    # knobs
    save_output = True
    # Files to save
    group_outcome_df = pd.DataFrame()
    regret_mse_df = pd.DataFrame()

    for ite in range(num_ite):
        print("------------------ ITERATION ", ite, "------------------")
        for arm in range(num_arms):
            outcome_lis_of_lis.append(np.random.normal(loc=arm_means[arm],
                                                       scale=sqrt(
                                                           arm_vars[arm]),
                                                       size=size))

        for chi in range(1,7):
            chi = chi/3
            ucb_inf_eps_bandit = Bandit(name='ucb_inf_eps', num_arms=num_arms,
                                        trt_dist_list=outcome_lis_of_lis)

            ucb_inf_eps_group, ucb_inf_eps_outcome = ucb_inf_eps(
                ucb_inf_eps_bandit,
                num_subjects,
                chi=chi)

            sub_df = create_group_dict(ucb_inf_eps_group, ucb_inf_eps_outcome,
                                       'ucb_inf_eps', ite, chi)
            group_outcome_df = group_outcome_df.append(sub_df)

            ucb_inf_eps_regret = regret_outcome(ucb_inf_eps_group,
                                                ucb_inf_eps_outcome)
            ucb_inf_eps_mean_mse, ucb_inf_eps_var_mse = mse_outcome(
                ucb_inf_eps_group,
                ucb_inf_eps_outcome,
                arm_means,
                arm_vars)
            sub_df = create_regret_dict(ucb_inf_eps_regret,
                                        ucb_inf_eps_mean_mse,
                                        ucb_inf_eps_var_mse,
                                        'ucb_inf_eps', ite, chi)
            regret_mse_df = regret_mse_df.append(sub_df)

        if save_output:
            group_outcome_df.to_csv("group_eps_sim.csv", index=False)
            regret_mse_df.to_csv("regret_mse_eps_sim.csv", index=False)


if __name__ == '__main__':
    main()
