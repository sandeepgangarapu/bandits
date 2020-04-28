from bandits.algorithms.ab_testing import ab_testing
from bandits.algorithms.ucb_inf import ucb_inf
from bandits.algorithms.thomp_inf_eps import thomp_inf_eps
from bandits.algorithms.thompson_sampling import thompson_sampling
from bandits.algorithms.ucb_inf_eps import ucb_inf_eps
from bandits.algorithms.ucb import ucb
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.bandit import Bandit
from bandits.utils import mse_outcome, prop_mse
from bandits.algorithms.weighed_estimators.weighed_estimators import ipw, aipw
from math import sqrt
import pandas as pd
import numpy as np


class BanditSimulation:

    def __init__(self, seed, num_ite, arm_means, arm_vars, eps_inf, horizon, alg_list, xi=1, output_file_path=None,
                 prop_file_path=None):
        """This class is run bandits simulation for given params and give simulation output.
        :param seed: seed for randomization
        :param num_ite: number of iterations
        :param arm_means: list of arm means
        :param arm_vars: list of arm vars
        :param eps_inf: epsilon (% exploration) for variance based allocation
        :param horizon: total number of available subjects
        :param alg_list: list of algorithms for simulations to run
        :param xi = xi value for inf eps
        """
        self.seed = seed
        self.num_ite = num_ite
        self.arm_means = arm_means
        self.arm_vars = arm_vars
        self.num_arms = len(arm_means)
        self.eps_inf = eps_inf
        self.horizon = horizon
        self.alg_list = alg_list
        self.xi = xi
        self.output_file_path = output_file_path
        self.prop_file_path = prop_file_path

    def run_simulation(self):

        outcome_lis_of_lis = self.generate_empirical_arm_outcome_dist()
        output_df_lis = []
        output_prop_lis = []

        for ite in range(self.num_ite):
            print("------------------RUNNING ITERATION ", ite, "------------------")
            bandit_dict = self.create_bandit_instances(outcome_lis_of_lis)
            # simulate all_algs
            for alg in self.alg_list:
                self.simulate_bandit(alg, bandit_dict[alg])

            # create df with necessary info

            for alg in self.alg_list:
                df = self.create_output_df(bandit_dict[alg], ite)
                output_df_lis.append(df)
                if 'thomp' in alg:
                    alg_name = alg+'_ipw'
                    ipw_est = ipw(bandit_dict[alg].arm_pull_tracker,
                                  bandit_dict[alg].reward_tracker,
                                  bandit_dict[alg].propensity_tracker)
                    df = self.create_prop_df(bandit_dict[alg], ite, ipw_est, alg_name)
                    output_prop_lis.append(df)
                    alg_name = alg + '_aipw'
                    aipw_est = aipw(bandit_dict[alg].arm_pull_tracker,
                                  bandit_dict[alg].reward_tracker,
                                  bandit_dict[alg].propensity_tracker)
                    df = self.create_prop_df(bandit_dict[alg], ite, aipw_est, alg_name)
                    output_prop_lis.append(df)

        output_df = pd.concat(output_df_lis)
        prop_df = pd.concat(output_prop_lis)
        # save output or return it
        if self.output_file_path is not None:
            output_df.to_csv(self.output_file_path, index=False)
            prop_df.to_csv(self.prop_file_path, index=False)
        else:
            return output_df, prop_df

    def generate_empirical_arm_outcome_dist(self):
        outcome_lis_of_lis = []
        for arm in range(self.num_arms):
            outcome_lis_of_lis.append(np.random.normal(loc=self.arm_means[arm],
                                                       scale=sqrt(self.arm_vars[arm]),
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
            ab_testing(bandit, self.horizon, self.arm_means, self.arm_vars, sample_size=None, post_allocation=True)
        if alg == 'ucb':
            ucb(bandit, self.horizon)
        if alg == 'eps_greedy':
            epsilon_greedy(bandit, self.horizon, epsilon=self.eps_inf)
        if alg == 'ucb_inf':
            ucb_inf(bandit, self.horizon, eps=self.eps_inf)
        if alg == 'ucb_inf_eps':
            ucb_inf_eps(bandit, self.horizon, xi=self.xi)
        if alg == 'thomp':
            thompson_sampling(bandit, self.horizon)
        if alg == 'thomp_inf_eps':
            thomp_inf_eps(bandit, self.horizon, xi=self.xi)

    def create_output_df(self, bandit, ite):
        mean_mse, var_mse = mse_outcome(bandit.arm_tracker, bandit.reward_tracker, self.arm_means, self.arm_vars)
        dict = {"group": bandit.arm_tracker,
                'outcome': bandit.reward_tracker,
                'regret': bandit.regret,
                'mean_mse': mean_mse,
                'var_mse': var_mse,
                'alg': bandit.name,
                'ite': ite}
        df = pd.DataFrame(dict)
        return df

    def create_prop_df(self, bandit, ite, mean_est, alg_name):
        mean_mse = prop_mse(bandit.arm_tracker, mean_est, self.arm_means)
        dict = {'mse': mean_mse,
                'alg': alg_name,
                'ite': ite,
                'mean_est': mean_est,
                'group': bandit.arm_tracker}
        df = pd.DataFrame(dict)
        return df

