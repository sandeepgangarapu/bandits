from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np



if __name__ == '__main__':
    true_means = [0.25, 1.82, 1.48, 2.25, 2]
    true_vars = [2.84,  1.97, 2.62, 1, 2.06]

    alg_list=['thomp_inf', 'thomp']
    #
    # estimator_list=['aipw', 'eval_aipw', 'ipw'],

    sim = BanditSimulation(num_ite=100, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=500,
                           alg_list=alg_list,
                           estimator_list=['aipw', 'ipw'],
                           mse_calc=False,
                           output_file_path='analysis/output/sim_ipw_100_500'
                                            '.csv')

    sim.run_simulation_multiprocessing()

