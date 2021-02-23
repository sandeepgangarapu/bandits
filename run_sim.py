from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time
import pandas as pd


def run_sim(file_path, true_means, true_vars=None, dist_type='Normal', xi=0.05):
    start_time = time.time()
    alg_list=['ab_bern', 'thomp_bern', 'thomp_athey_bern', 'thomp_inf_bern']
    alg_list = ['thomp_inf_bern']

    estimator_list=['aipw', 'eval_aipw', 'ipw']

    sim = BanditSimulation(num_ite=31, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.2,
                           horizon=10000,
                           alg_list=alg_list,
                           estimator_list=None,
                           mse_calc=True,
                           agg=False,
                           xi=xi,
                           cap_prop=True,
                           dist_type=dist_type,
                           output_file_path=file_path)
    sim.run_simulation_multiprocessing()
    print("--- %s mins ---" % ((time.time() - start_time)/60))
    return true_means, true_vars

if __name__ == '__main__':
    meta_analysis = False
    normal_analysis = False
    regret_order = False
    xi_analysis = True
    bernoulli_analysis = False
    hsn = False
    lsn = False
    zsn = False
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
        true_means = [0.25, 1.82, 1.48, 2.25, 2]
        true_vars = [2.84, 1.97, 2.62, 1, 2.06]
        a = run_sim('analysis/output/agg_analysis_default_100_20000.csv', true_means, true_vars=true_vars, dist_type='Normal')
    if bernoulli_analysis:
        true_means = np.array([0.157, 0.178, 0.199, 0.146, 0.129])
        true_vars = true_means*(1-true_means)
        run_sim('analysis/output/non_agg_bern_62_20000.csv', true_means, true_vars=true_vars, dist_type='Bernoulli')
    if lsn:
        true_means = [1, 1.1, 1.2]
        true_vars = [1/3, 1/3, 1/3]
        run_sim('analysis/output/non_agg_hsn_100_20000.csv', true_means, true_vars=true_vars, dist_type='HSN')
    if hsn:
        true_means = [1, 1.5, 2]
        true_vars = [1/3, 1/3, 1/3]
        run_sim('analysis/output/non_agg_lsn_100_20000.csv', true_means, true_vars=true_vars, dist_type='LSN')
    if zsn:
        true_means = [1, 1, 1]
        true_vars = [1/3, 1/3, 1/3]
        run_sim('analysis/output/non_agg_zsn_100_20000.csv', true_means, true_vars=true_vars, dist_type='ZSN')
    if regret_order:
        true_means = [0.25, 1.82, 1.48, 2.25, 2]
        true_vars = [2.84, 1.97, 2.62, 1, 2.06]
        a = run_sim('analysis/output/wise_regret_analysis_100.csv', true_means, true_vars=true_vars, dist_type='Normal')
    if xi_analysis:
        true_means = np.array([0.157, 0.178, 0.199, 0.146, 0.129])
        true_vars = true_means * (1 - true_means)
        xi = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.1, 1.5]
        for x in xi:
            f_path = 'analysis/output/xi_analysis' + str(x) + '.csv'
            run_sim(f_path, true_means, true_vars=true_vars, dist_type='Bernoulli', xi=x)
