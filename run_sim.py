from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time
import pandas as pd


def run_sim(file_path, true_means, true_vars=None, dist_type='Normal'):
    start_time = time.time()
    alg_list=['ab', 'thomp', 'thomp_inf']

    estimator_list=['aipw', 'eval_aipw', 'ipw']

    sim = BanditSimulation(num_ite=31, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=2000,
                           alg_list=alg_list,
                           estimator_list=None,
                           mse_calc=True,
                           agg=False,
                           xi=0.8,
                           cap_prop=True,
                           dist_type=dist_type,
                           output_file_path=file_path)
    sim.run_simulation_multiprocessing()
    print("--- %s seconds ---" % (time.time() - start_time))
    return true_means, true_vars

if __name__ == '__main__':
    meta_analysis = False
    normal_analysis = True
    # if meta_analysis:
    #     num_meta_ite = 100
    #     ref_lis = []
    #     for i in range(num_meta_ite):
    #         f_path = 'analysis/output/sim_weighed_ite_1000_t_500_metaite_' + str(i) + '.csv'
    #         mn, vr = run_sim(f_path)
    #         ref = pd.DataFrame({'mn':mn, 'vr':vr, 'ite': np.repeat(i, 5)})
    #         ref_lis.append(ref)
    #         final_out = pd.concat(ref_lis)
    #         final_out.to_csv("analysis/output/ref.csv", index=False)
    if normal_analysis:
        true_means = [0.25, 1.82, 1.48, 2.25, 2]
        true_vars = [2.84, 1.97, 2.62, 1, 2.06]
        a = run_sim('analysis/output/wise_mse_graph.csv', true_means, true_vars=true_vars, dist_type='Normal')
