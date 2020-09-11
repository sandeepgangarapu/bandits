from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time

if __name__ == '__main__':
    start_time = time.time()
    true_means = np.random.uniform(0,3,5)
    true_vars = np.random.uniform(0,3,5)
    print(true_means)
    print(true_vars)
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
                           output_file_path='analysis/output/sim_weighed_1000_500_2'
                                            '.csv')

    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
