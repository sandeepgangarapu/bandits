from algorithms.ab_testing import ab_testing
from algorithms.thompson_sampling import thompson_sampling
from algorithms.thomp_bern import thompson_sampling_bern
from algorithms.thomp_bern_batched import thompson_sampling_bern_batched
from algorithms.thompson_inf import thomp_inf
from algorithms.thompson_inf_bern import thomp_inf_bern
from algorithms.thompson_inf_bern_batched import thomp_inf_bern_batched
from algorithms.ab_testing_bern import ab_testing_bern
from algorithms.ucb_inf_eps import ucb_inf_eps
from algorithms.ucb import ucb
from algorithms.epsilongreedy import epsilon_greedy
from bandit import Bandit
from utils import mse_outcome, prop_mse
from algorithms.weighed_estimators import weighed_estimators
import pandas as pd
import multiprocessing
import numpy as np


class BanditSimulation:

    def __init__(self, num_ite, arm_means, eps_inf, horizon, alg_list, arm_vars=None, mse_calc=True,
                 agg=False, estimator_list=None, xi=1, dist_type='Normal',
                 cap_prop=True, output_file_path=None, post_allocation=True):
        """This class is to run bandits simulation for given params and give simulation output.
        :param seed: seed for randomization
        :param num_ite: number of iterations
        :param arm_means: list of arm means
        :param arm_vars: list of arm vars
        :param eps_inf: epsilon (% exploration) for variance based allocation
        :param horizon: total number of available subjects
        :param alg_list: list of algorithms for simulations to run
        :param mse_calc: whether to calculate MSE
        :param agg: output aggregated values like mean etc instead of raw values of rewards
        :param estimator_list: list of estimators like ipw, aipw etc
        :param xi: xi value for inf eps
        :param dist_type: type of outcome distribution
        :param cap_prop: capping the propensity in thompson based algs
        :param output_file_path: path of file where output needs to be stored
        """
        self.num_ite = num_ite
        self.arm_means = arm_means
        self.arm_vars = arm_vars
        self.num_arms = len(arm_means)
        self.eps_inf = eps_inf
        self.horizon = horizon
        self.alg_list = alg_list
        self.mse_calc = mse_calc
        self.agg = agg
        self.estimator_list = estimator_list
        self.xi = xi
        self.dist_type = dist_type
        self.output_file_path = output_file_path
        self.cap_prop = cap_prop
        self.type_of_pull = 'monte_carlo' if self.estimator_list else 'single'
        self.post_allocation = post_allocation
        # self.outcome_lis_of_lis = self.generate_empirical_arm_outcome_dist()

    def run_simulation_multiprocessing(self):
        """ This wraps around the rest of the functioning so that parallel processing happens.
        :return: either saves the output or gives out data frame of output
        """
        a_pool = multiprocessing.Pool()
        print(multiprocessing.cpu_count())
        result = a_pool.map(self.run_simulation, range(self.num_ite))
        # save output or return it
        final_output = pd.concat(result)
        if self.output_file_path is not None:
            final_output.to_csv(self.output_file_path, index=False)

        if self.output_file_path is not None:
            final_output.to_csv(self.output_file_path, index=False)
        else:
            return final_output

    def run_simulation(self, ite):
        """
        each simulation is run in this method
        :param ite: number of ite so it can be stored in the output
        :return: the output of the iteration
        """
        # we do this so that each process spawns at different random seed
        np.random.seed(ite)
        # list of outputs of all algorithms for a given iteration
        output_df_lis = []
        # list of outputs of estimators of all algs for a given iteration
        output_prop_lis = []
        print("------------------RUNNING ITERATION ", ite, "------------------")
        bandit_dict = self.create_bandit_instances()
        # simulate all_algs
        for alg in self.alg_list:
            self.simulate_bandit(alg, bandit_dict[alg])
        # create df with necessary info
        for alg in self.alg_list:
            df = self.create_output_df(bandit_dict[alg], ite, self.mse_calc)
            output_df_lis.append(df)
            # for each of the algorithms we see if any estimator needs to be calculated
            if self.estimator_list:
                if 'thomp' in alg:
                    for est in self.estimator_list:
                        df = self.run_estimator(alg, est, bandit_dict[alg], ite)
                        output_prop_lis.append(df)
        output_df = pd.concat(output_df_lis)
        if output_prop_lis:
            prop_df = pd.concat(output_prop_lis)
            ite_output = pd.concat([output_df, prop_df], axis=0, ignore_index=True)
        else:
            ite_output = output_df
        return ite_output

    def run_estimator(self, alg, estimator_name, bandit, ite):
        """
        This method will simulate estimators
        :param alg: algorithm name
        :param estimator_name: estimator name
        :param bandit: bandit instance
        :param ite: num of ite
        :return: returns the df of the output of the estimator + alg + ite combination
        """
        alg_name = alg + "_" + estimator_name
        est_value = weighed_estimators(estimator_name,
                                       bandit.arm_tracker,
                                       bandit.reward_tracker,
                                       bandit.propensity_tracker,
                                       weight_lis_of_lis=bandit.prop_lis_tracker)
        df = self.create_prop_df(bandit, ite, est_value, alg_name, self.mse_calc)
        return df

    def create_bandit_instances(self):
        """
        This method will create an instance for each bandit we simulate
        :return: dict of bandit instances
        """
        bandit_dict = {}
        for alg in self.alg_list:
            bandit_name = alg
            bandit_dict[bandit_name] = Bandit(name=alg, arm_means=self.arm_means, arm_vars=self.arm_vars,
                                              dist_type=self.dist_type)
        return bandit_dict

    def simulate_bandit(self, alg, bandit):
        """
        This method will simulate the bandit and store the values of simulations in the instance in place
        :param alg: name of the algorithm
        :param bandit: instance of the bandit
        :return: None
        """
        if alg == 'ab':
            sample_size = int(self.horizon/self.num_arms) if not self.post_allocation else None
            ab_testing(bandit, self.horizon, sample_size=sample_size, post_allocation=self.post_allocation)
        if alg == 'ucb':
            ucb(bandit, self.horizon, type_of_pull=self.type_of_pull)
        if alg == 'eps_greedy':
            epsilon_greedy(bandit, self.horizon, epsilon=self.eps_inf)
        if alg == 'ucb_inf_eps':
            ucb_inf_eps(bandit, self.horizon, xi=self.xi, type_of_pull=self.type_of_pull)
        if alg == 'thomp':
            thompson_sampling(bandit, self.horizon, type_of_pull=self.type_of_pull, cap_prop=False)
        if alg == 'thomp_athey':
            thompson_sampling(bandit, self.horizon, type_of_pull=self.type_of_pull, cap_prop=self.cap_prop)
        if alg == 'thomp_inf':
            thomp_inf(bandit, self.horizon, xi=self.xi, type_of_pull=self.type_of_pull, cap_prop=self.cap_prop)
        if alg == 'thomp_bern':
            thompson_sampling_bern(bandit, self.horizon, type_of_pull=self.type_of_pull, cap_prop=False)
        if alg == 'thomp_bern_batched':
            thompson_sampling_bern_batched(bandit, self.horizon, cap_prop=False)
        if alg == 'thomp_athey_bern':
            thompson_sampling_bern(bandit, self.horizon, type_of_pull=self.type_of_pull, cap_prop=self.cap_prop)
        if alg == 'thomp_inf_bern':
            thomp_inf_bern(bandit, self.horizon, xi=self.xi, type_of_pull=self.type_of_pull, cap_prop=self.cap_prop)
        if alg == 'thomp_inf_bern_batched':
            thomp_inf_bern_batched(bandit, self.horizon, xi=self.xi,
                                   cap_prop=self.cap_prop)
        if alg == 'ab_bern':
            sample_size = int(self.horizon / self.num_arms) if not self.post_allocation else None
            ab_testing_bern(bandit, self.horizon, sample_size=sample_size, post_allocation=self.post_allocation)

    def create_output_df(self, bandit, ite, mse_calc):
        """
        This method will create data frame of output for the iteration and a bandit and return it
        :param bandit: bandit instance
        :param ite: number of ite
        :param mse_calc: whether to calculate mse
        :return: data frame of the output
        """
        if self.agg:
            dict_df = {'group': list(range(self.num_arms)),
                       'mean_est': bandit.avg_reward_tracker,
                       'var_est' : bandit.var_est_tracker,
                       'arm_pull_tracker': bandit.arm_pull_tracker,
                       'alg': bandit.name,
                       'ite': ite}
            df = pd.DataFrame(dict_df)
            # TODO add mse calc when aggregation is true
        else:
            dict_df = {"group": bandit.arm_tracker,
                       'outcome': bandit.reward_tracker,
                       'regret': bandit.regret,
                       'alg': bandit.name,
                       'ite': ite}
            if mse_calc:
                mean_mse, var_mse = mse_outcome(bandit.arm_tracker, bandit.reward_tracker, self.arm_means, self.arm_vars)
                dict_df['mean_mse'] = mean_mse
                dict_df['var_mse'] = var_mse
            df = pd.DataFrame(dict_df)
        return df

    def create_prop_df(self, bandit, ite, mean_est, alg_name, mse_calc):
        """
        This method will create data frame of output for the estimators of a bandit and return it
        :param bandit: bandit instance
        :param ite: num of ite
        :param mean_est: list of mean_est calculated at every time period
        :param alg_name: name of alg + estimator name (ex: thomp_aipw)
        :param mse_calc: Bool to calculate MSE
        :return: output df
        """
        # var est here is just a dummy to enable the rowbinding of normal and prop df's
        if self.agg:
            dict_df = {'alg': alg_name,
                       'ite': ite,
                       'group': list(range(self.num_arms)),
                       'arm_pull_tracker': bandit.arm_pull_tracker,
                       'mean_est': mean_est,
                       'var_est': mean_est}
            df = pd.DataFrame(dict_df)
            # TODO add mse calc when aggregation is true
        else:
            dict_df = {'alg': alg_name,
                       'ite': ite,
                       'group': bandit.arm_tracker,
                       'mean_est': mean_est}
            if mse_calc:
                mean_mse = prop_mse(bandit.arm_tracker, mean_est, self.arm_means)
                dict_df['mse'] = mean_mse
            df = pd.DataFrame(dict_df)
        return df
