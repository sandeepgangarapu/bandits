from bandits.bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import pandas as pd

num_sims = 10
num_groups = 5
horizon = 1000
max_mean = 2
max_var = 3
main_df_list = []
mean_var_df_list = []
arm_means = np.random.uniform(0, max_mean, num_groups)
arm_vars = np.random.uniform(0, max_var, num_groups)
print(arm_means, arm_vars)

for subsim in range(num_sims):
    
    sim = BanditSimulation(seed=subsim, num_ite=1, arm_means=arm_means,
                           arm_vars=arm_vars,
                           eps_inf=0.2,
                           horizon=horizon,
                           alg_list=['ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
                                     'thomp', 'thomp_inf_eps'],
                          # estimator_list=['ipw', 'aipw', 'eval_aipw'],
                           mse_calc=False)
    df = sim.run_simulation()
    df['sim'] = np.repeat(subsim, df.shape[0])
    main_df_list.append(df)
    mean_df = pd.DataFrame({'mean':arm_means, 'vars':arm_vars,
                            'group': [i for i in range(num_groups)]})
    mean_df['sim'] = np.repeat(subsim, mean_df.shape[0])
    mean_var_df_list.append(mean_df)

main_output = pd.concat(main_df_list, axis=0, ignore_index=True).to_csv(
    'analysis/output/hypo.csv', index=False)
mean_output = pd.concat(mean_var_df_list, axis=0, ignore_index=True).to_csv(
    'analysis/output/hypo_mean.csv', index=False)
