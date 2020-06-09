from bandits.bandit_comparison_simulation import BanditSimulation
import numpy as np
#
# true_means = [0.25, 1.82, 1.48, 2.25, 2]
# true_vars = [2.84,  1.97, 2.62, 1, 2.06]


true_means = [2, 2, 2]
true_vars = [1, 2.06, 0.2]


sim = BanditSimulation(seed=583257, num_ite=50, arm_means=true_means,
                       arm_vars=true_vars,
                       eps_inf=0.2,
                       horizon=500,
                       alg_list=['ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
                                     'thomp', 'thomp_inf_eps'],
                       estimator_list=['aipw', 'eval_aipw', 'ipw'],
                       output_file_path='analysis/output/sim_hypo_same_mean_diff_var'
                                        '.csv')
sim.run_simulation()


true_means = [2, 2, 2]
true_vars = [1, 1, 1]


sim = BanditSimulation(seed=583257, num_ite=50, arm_means=true_means,
                       arm_vars=true_vars,
                       eps_inf=0.2,
                       horizon=500,
                       alg_list=['ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
                                     'thomp', 'thomp_inf_eps'],
                       estimator_list=['aipw', 'eval_aipw', 'ipw'],
                       output_file_path='analysis/output/sim_hypo_same_mean_same_var'
                                        '.csv')
sim.run_simulation()

