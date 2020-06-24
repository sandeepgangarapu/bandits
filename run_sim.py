import sys
from bandits.bandit_comparison_simulation import BanditSimulation
import numpy as np





true_means = [0.1, 6.8, 7]
true_vars = [3, 1, 1]

true_means = [0.25, 2.3, 2]
true_vars = [2.84, 1, 2.06]

true_means = [1.5, 2.8, 3]
true_vars = [1, 1, 1]

true_means = [0.25, 1.82, 1.48, 2.25, 2]
true_vars = [2.84,  1.97, 2.62, 1, 2.06]

# alg_list=['ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
#                                      'thomp', 'thomp_inf_eps'],
#
# estimator_list=['aipw', 'eval_aipw', 'ipw'],

sim = BanditSimulation(seed=12412, num_ite=3, arm_means=true_means,
                       arm_vars=true_vars,
                       eps_inf=0.2,
                       horizon=2000,
                       alg_list=['ab', 'thomp', 'thomp_inf_eps'],
                       # estimator_list=['aipw', 'ipw'],
                       mse_calc = True,
                       output_file_path='analysis/output/regret_3_2000_1'
                                        '.csv')

sim.run_simulation()
