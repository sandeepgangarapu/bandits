import numpy as np
from utils import thompson_arm_pull, bayesian_update_normal_inv_gamma, trt_dist_list
from bandit import Bandit

def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
        bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms * xi)
    eps_n = eta / (1 + eta)
    return eps_n


def thomp_inf(bandit, num_rounds, xi=1, type_of_pull='single'):
    print("---------------Running Thompson Sampling INF EPS ---------------")

    # allocate one subject to each arm (We can remove this later rules)
    for ite in range(2):
        for arm in range(bandit.num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)
    num_arms = bandit.num_arms

    # prior = NormalGamma(m0, n0, s0^2, d0)
    # m0 = mean, n0=num_values, s0^2=sample variance, d0=n0-1(deg of freedom)
    prior_params = [(bandit.avg_reward_tracker[i], 2, bandit.var_est_tracker[i], 1) for i in
                    range(num_arms)]
    
    # for now, lets assume all groups have same no. of subjects
    for subject in range(int((num_rounds-(2*num_arms))/2)):
        # we divide by 2 because we want atleast 2 values to update prior
        # perc_ab of the time, we do variance based, otherwise thompson
        if np.random.uniform(0, 1) < calc_eps_n(bandit, xi):
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = \
                np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)
            # normalizing std.err square for calculating probabilities
            norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
            arm_var = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm_var, prop_lis=norm_prob)
                bandit.pull_arm(arm_var, prop_lis=norm_prob)
            else:
                bandit.pull_arm(arm_var)
                bandit.pull_arm(arm_var)
            reward_1 = bandit.reward_tracker[-1]
            reward_2 = bandit.reward_tracker[-2]
            # posterior distribution is estimated using bayesian update
            prior_params[arm_var] = bayesian_update_normal_inv_gamma(prior_params[arm_var],
                                                                     lis_x=[reward_1, reward_2])
        else:
            arm_means = [i[0] for i in prior_params]
            arm_vars = [i[2] for i in prior_params]
            if type_of_pull == 'monte_carlo':
                chosen_arm, prop_lis = thompson_arm_pull(mean_lis=arm_means,
                                                         var_lis=arm_vars,
                                                         type_of_pull=type_of_pull)
            else:
                chosen_arm = thompson_arm_pull(mean_lis=arm_means,
                                               var_lis=arm_vars,
                                               type_of_pull=type_of_pull)
            # That chosen arm is pulled to observe reward
            # we pull the arm twice
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
                bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
            else:
                bandit.pull_arm(chosen_arm)
                bandit.pull_arm(chosen_arm)

            reward_1 = bandit.reward_tracker[-1]
            reward_2 = bandit.reward_tracker[-2]
            # posterior distribution is estimated using bayesian update
            prior_params[chosen_arm] = bayesian_update_normal_inv_gamma(prior_params[chosen_arm],
                                                                        lis_x=[reward_1, reward_2])
    return bandit
