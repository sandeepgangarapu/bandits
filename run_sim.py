from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time
import pandas as pd


def run_sim(file_path, true_means, true_vars):
    start_time = time.time()
    alg_list=['thomp', 'thomp_inf']

    estimator_list=['aipw', 'eval_aipw', 'ipw']

    sim = BanditSimulation(num_ite=1000, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=20000,
                           alg_list=alg_list,
                           estimator_list=estimator_list,
                           type_of_eval_weight='constant_allocation',
                           mse_calc=False,
                           agg=True,
                           xi=0.8,
                           cap_prop=True,
                           output_file_path=file_path)
    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
    return true_means, true_vars

if __name__ == '__main__':
    meta_analysis = False
    normal_analysis = True
    if meta_analysis:
        num_meta_ite = 100
        ref_lis = []
        for i in range(num_meta_ite):
            f_path = 'analysis/output/sim_weighed_ite_1000_t_500_metaite_' + str(i) + '.csv'
            mn, vr = run_sim(f_path)
            ref = pd.DataFrame({'mn':mn, 'vr':vr, 'ite': np.repeat(i, 5)})
            ref_lis.append(ref)
            final_out = pd.concat(ref_lis)
            final_out.to_csv("analysis/output/ref.csv", index=False)
    if normal_analysis:
        true_means = [1, 2, 3]
        true_vars = [1, 1, 1]
        a = run_sim('analysis/output/athey_ite_1000_t_20000.csv', true_means, true_vars)
