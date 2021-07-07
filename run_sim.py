from bandit_comparison_simulation_raw import BanditSimulation
import numpy as np
import time
import pandas as pd


def run_sim(file_path, true_means, true_vars=None, dist_type='Normal',
            xi=0.005):
    start_time = time.time()
    #alg_list=['thomp_bern_batched', 'thomp_inf_bern_batched', 'thomp_bern',
    #          'thomp_inf_bern', 'ab_bern']
    # alg_list = ['thomp_inf_bern_batched']
    alg_list=['thomp_inf']
    estimator_list = ['eval_aipw']

    sim = BanditSimulation(num_ite=1280, arm_means=true_means,
                           arm_vars=true_vars,
                           eps_inf=0.5,
                           horizon=2000,
                           alg_list=alg_list,
                           estimator_list=estimator_list,
                           mse_calc=False,
                           agg=False,
                           final_metrics=True,
                           xi=xi,
                           batch_size=100,
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
    xi_analysis = False
    bernoulli_analysis = False
    hsn = False
    lsn = False
    zsn = False
    hsn_bern = False
    lsn_bern = False
    zsn_bern = False
    xi_analysis_hsn = False
    xi_analysis_lsn = False
    xi_analysis_zsn = True
    xi = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8,
          1.1, 1.5]
    true_vars = [1/3, 1/3, 1/3]
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
        true_means = np.array([0.37098621, 0.33080171, 0.1699615, 0.18902466, 0.6743146])
        true_vars = true_means*(1-true_means)
        run_sim('analysis/output/agg_analysis_batched.csv', true_means,
                true_vars=true_vars, dist_type='Bernoulli')
    if lsn:
        true_means = [1, 1.1, 1.2]
        run_sim('analysis/output/non_agg_hsn_100_20000.csv', true_means,
                true_vars=true_vars, dist_type='LSN')
    if hsn:
        true_means = [1, 1.5, 2]
        run_sim('analysis/output/batched_test.csv', true_means,
                true_vars=true_vars, dist_type='HSN')
    if zsn:
        true_means = [1, 1, 1]
        run_sim('analysis/output/non_agg_zsn_100_20000.csv', true_means, true_vars=true_vars, dist_type='ZSN')
    if lsn_bern:
        true_means = [0.45, 0.5, 0.55]
        run_sim('analysis/output/lsn_bern_agg_1280_1000.csv', true_means,
                dist_type='LSN_bern')
    if hsn_bern:
        true_means = [0.25, 0.5, 0.75]
        run_sim('analysis/output/hsn_bern_agg_1280_1000.csv', true_means,
                dist_type='HSN_bern')
    if zsn_bern:
        true_means = [0.5, 0.5, 0.5]
        run_sim('analysis/output/zsn_bern_agg_1280_1000.csv', true_means,
                dist_type='ZSN_bern')
    if regret_order:
        true_means = [0.25, 1.82, 1.48, 2.25, 2]
        true_vars = [2.84, 1.97, 2.62, 1, 2.06]
        a = run_sim('analysis/output/wise_regret_analysis_100.csv', true_means, true_vars=true_vars, dist_type='Normal')
    if xi_analysis_lsn:
        true_means = [1, 1.1, 1.2]
        for x in xi:
            f_path = 'analysis/output/xi_analysis' + str(x) + '_lsn.csv'
            run_sim(f_path, true_means,
                    true_vars=true_vars, dist_type='LSN', xi=x)
    if xi_analysis_hsn:
        true_means = [1, 1.5, 2]
        for x in xi:
            f_path = 'analysis/output/xi_analysis' + str(x) + '_hsn.csv'
            run_sim(f_path, true_means,
                    true_vars=true_vars, dist_type='HSN', xi=x)
    if xi_analysis_zsn:
        true_means = [1, 1, 1]
        for x in xi:
            f_path = 'analysis/output/xi_analysis' + str(x) + '_zsn.csv'
            run_sim(f_path, true_means,
                    true_vars=true_vars, dist_type='ZSN', xi=x)