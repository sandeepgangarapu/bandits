from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import pandas as pd
import time
from statsmodels.stats.power import TTestIndPower


def run_sim(file_path, true_means, true_vars=None, horizon=None, num_ite=31, dist_type='Normal'):
    start_time = time.time()
    alg_list=['ab', 'thomp', 'thomp_inf']
    estimator_list=['eval_aipw', 'eval_aipw_var']
    sim = BanditSimulation(num_ite=num_ite,
                           arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=horizon,
                           alg_list=alg_list,
                           estimator_list=estimator_list,
                           mse_calc=False,
                           agg=True,
                           xi=0.8,
                           cap_prop=True,
                           dist_type=dist_type,
                           output_file_path=file_path,
                           post_allocation=False)
    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
    return true_means, true_vars

if __name__ == '__main__':
    effect_sizes = [0.2, 0.5, 0.8]
    alpha = 0.05
    beta = 0.1
    sample_sizes = []
    for effect in effect_sizes:
        sample_size = TTestIndPower().solve_power(effect_size=effect, power=1-beta, alpha=alpha)
        sample_sizes.append(int(sample_size))
    # true_vars = [1, 1]
    # for i in range(len(sample_sizes)):
        # true_means = [0]
        # true_means.append(effect_sizes[i])
    true_means = [0.25, 1.82, 1.48, 2.25, 2]
    true_vars = [2.84, 1.97, 2.62, 1, 2.06]
    # horizon = len(true_means) * sample_sizes[i]
    horizon = 2000
    num_ite = 1000
    file_name = 'analysis/output/hyp_ite_'+ str(num_ite) + '_t_'+ str(horizon) + '.csv'
    a = run_sim(file_name, true_means, true_vars=true_vars, num_ite=num_ite, horizon=horizon)
