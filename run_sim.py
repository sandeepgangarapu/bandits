from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time

if __name__ == '__main__':
    start_time = time.time()
    true_means = [0.25, 1.82, 1.48, 2.25, 2]
    true_vars = [2.84,  1.97, 2.62, 1, 2.06]

    alg_list=['thomp', 'thomp_inf']

    estimator_list=['aipw', 'eval_aipw', 'ipw']

    sim = BanditSimulation(num_ite=1000, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=500,
                           alg_list=alg_list,
                           estimator_list=estimator_list,
                           type_of_eval_weight='constant_allocation',
                           mse_calc=False,
                           agg=True,
                           xi=0.8,
                           output_file_path='analysis/output/sim_weighed_1000_500'
                                            '.csv')

    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
