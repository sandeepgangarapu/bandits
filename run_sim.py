from bandits.bandit_comparison_simulation import BanditSimulation
import numpy as np


true_means = [0.25, 1.82, 1.48, 2.25, 2]
true_vars = [2.84,  1.97, 2.62, 1, 2.06]


# true_means = [0.25, 2.25, 2]
# true_vars = [2.84, 1, 2.06]

# alg_list=['ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
#                                      'thomp', 'thomp_inf_eps'],
#
# estimator_list=['aipw', 'eval_aipw', 'ipw'],

sim = BanditSimulation(seed=583257, num_ite=300, arm_means=true_means,
                       arm_vars=true_vars,
                       eps_inf=0.2,
                       horizon=1000,
                       alg_list=['thomp', 'thomp_inf_eps', 'ucb',
                                 'ucb_inf_eps'],
                       estimator_list=['aipw', 'ipw', 'eval_aipw'],
                       mse_calc=False,
                       output_file_path='analysis/output/sim_thomp_ucb_300_1000'
                                        '.csv')
sim.run_simulation()
