from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time
import pandas as pd

def run_sim(file_path):
    start_time = time.time()
    true_means = np.random.uniform(0,3,5)
    true_vars = np.random.uniform(0,3,5)
    print(true_means)
    print(true_vars)
    alg_list=['thomp', 'thomp_inf']

    estimator_list=['aipw', 'eval_aipw', 'ipw']

    sim = BanditSimulation(num_ite=2, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=500,
                           alg_list=alg_list,
                           estimator_list=estimator_list,
                           type_of_eval_weight='constant_allocation',
                           mse_calc=False,
                           agg=True,
                           xi=0.8,
                           output_file_path=file_path)
    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
    return true_means, true_vars

if __name__ == '__main__':
    num_meta_ite = 2
    ref_lis = []
    for i in range(num_meta_ite):
        file_path = 'analysis/output/sim_weighed_ite_1000_t_500_metaite_' + str(i) + '.csv'
        mn, vr = run_sim(file_path)
        ref = pd.DataFrame({'mn':mn, 'vr':vr, 'ite': np.repeat(i, 5)})
        ref_lis.append(ref)
        final_out = pd.concat(ref_lis)
        final_out.to_csv("ref.csv")
