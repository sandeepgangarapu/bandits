from bandits.algorithms.ab_testing import ab_testing
from bandits.algorithms.thomp_inf_eps import thomp_inf_eps
from bandits.algorithms.thompson_sampling import thompson_sampling
from bandits.algorithms.ucb_inf_eps import ucb_inf_eps
from bandits.algorithms.ucb import ucb
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.utils import mse_outcome, prop_mse
from bandits.algorithms.weighed_estimators.weighed_estimators import ipw, aipw, eval_aipw
from math import sqrt
import pandas as pd
import numpy as np
import multiprocessing

class BanditSimulation:

    def __init__(self, seed, num_ite, arm_means, arm_vars, eps_inf, horizon, alg_list, mse_calc=True,
                 estimator_list=None, xi=1, dist_type='Normal', output_file_path=None):
        """This class is run bandits simulation for given params and give simulation output.
        :param seed: seed for randomization
        :param num_ite: number of iterations
        :param arm_means: list of arm means
        :param arm_vars: list of arm vars
        :param eps_inf: epsilon (% exploration) for variance based allocation
        :param horizon: total number of available subjects
        :param alg_list: list of algorithms for simulations to run
        :param xi = xi value for inf eps
        :param dist_type = type of outcome distribution
        """
        self.seed = seed
        self.num_ite = num_ite
        self.arm_means = arm_means
        self.arm_vars = arm_vars
        self.num_arms = len(arm_means)
        self.eps_inf = eps_inf
        self.horizon = horizon
        self.alg_list = alg_list
        self.mse_calc = mse_calc
        self.estimator_list = estimator_list
        self.xi = xi
        self.dist_type = dist_type
        self.output_file_path = output_file_path
        self.type_of_pull = 'monte_carlo' if self.estimator_list else 'single'
        self.outcome_lis_of_lis = self.generate_empirical_arm_outcome_dist()

    def run_simulation_multiprocessing(self):
        a_pool = multiprocessing.Pool()
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
        output_df_lis = []
        output_prop_lis = []
        print("------------------RUNNING ITERATION ", ite, "------------------")
        bandit_dict = self.create_bandit_instances(self.outcome_lis_of_lis)
        # simulate all_algs
        for alg in self.alg_list:
            self.simulate_bandit(alg, bandit_dict[alg])

        # create df with necessary info

        for alg in self.alg_list:
            df = self.create_output_df(bandit_dict[alg], ite, self.mse_calc)
            output_df_lis.append(df)
            if self.estimator_list:
                if 'thomp' in alg:
                    for est in self.estimator_list:
                        df = self.run_estimators(alg, est, bandit_dict[alg], ite)
                        output_prop_lis.append(df)
                if 'ucb' in alg:
                    for est in self.estimator_list:
                        df = self.run_estimators(alg, est, bandit_dict[alg], ite)
                        output_prop_lis.append(df)
        output_df = pd.concat(output_df_lis)
        if output_prop_lis:
            prop_df = pd.concat(output_prop_lis)
            final_output = pd.concat([output_df, prop_df], axis=0, ignore_index=True)
        else:
            final_output = output_df
        return final_output



    def run_estimators(self, alg, estimator_name, bandit, ite):
        alg_name = alg + "_" + estimator_name
        if estimator_name == 'ipw':
            est_value = ipw(bandit.arm_tracker,
                            bandit.reward_tracker,
                            bandit.propensity_tracker)
        if estimator_name == 'aipw':
            est_value = aipw(bandit.arm_tracker,
                             bandit.reward_tracker,
                             bandit.propensity_tracker)
        if estimator_name == 'eval_aipw':
            est_value = eval_aipw(bandit.arm_tracker,
                                  bandit.reward_tracker,
                                  bandit.propensity_tracker,
                                  type_of_weight='variance_stabilizing',
                                  weight_lis_of_lis=bandit.prop_lis_tracker)

        df = self.create_prop_df(bandit, ite, est_value, alg_name, self.mse_calc)
        return df

    def generate_empirical_arm_outcome_dist(self):
        outcome_lis_of_lis = []
        for arm in range(self.num_arms):
            if self.dist_type == 'Normal':
                outcome_lis_of_lis.append(np.random.normal(loc=self.arm_means[arm],
                                                           scale=sqrt(self.arm_vars[arm]),
                                                           size=1000000))
            if self.dist_type == 'Bernoulli':
                outcome_lis_of_lis.append(np.random.binomial(size=100000, n=1, p=self.arm_means[arm]))
            if self.dist_type == 'Uniform':
                outcome_lis_of_lis.append(np.random.uniform(low=self.arm_means[arm]-2,
                                                            high=self.arm_means[arm]+2,
                                                            size=100000))
        return outcome_lis_of_lis

    def create_bandit_instances(self, outcome_lis_of_lis):
        bandit_dict = {}
        for alg in self.alg_list:
            bandit_name = alg
            bandit_dict[bandit_name] = Bandit(name=alg, num_arms=self.num_arms,
                                              trt_dist_list=outcome_lis_of_lis)
        return bandit_dict

    def simulate_bandit(self, alg, bandit):
        if alg == 'ab':
            ab_testing(bandit, self.horizon, sample_size=None, post_allocation=True)
        if alg == 'ucb':
            ucb(bandit, self.horizon, type_of_pull=self.type_of_pull)
        if alg == 'eps_greedy':
            epsilon_greedy(bandit, self.horizon, epsilon=self.eps_inf)
        if alg == 'ucb_inf_eps':
            ucb_inf_eps(bandit, self.horizon, xi=self.xi, type_of_pull=self.type_of_pull)
        if alg == 'thomp':
            thompson_sampling(bandit, self.horizon, type_of_pull=self.type_of_pull)
        if alg == 'thomp_inf_eps':
            thomp_inf_eps(bandit, self.horizon, xi=self.xi, type_of_pull=self.type_of_pull)

    def create_output_df(self, bandit, ite, mse_calc):
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
        dict_df = {'alg': alg_name,
                   'ite': ite,
                   'group': bandit.arm_tracker,
                   'mean_est' : mean_est}
        if mse_calc:
            mean_mse = prop_mse(bandit.arm_tracker, mean_est, self.arm_means)
            dict_df['mse'] = mean_mse
        df = pd.DataFrame(dict_df)
        return df

