from bandit import Bandit
import numpy as np
import random

def calc_eps_n(bandit, xi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms*xi)
    eps_n = eta/(1+eta)
    return eps_n

def epsilon_inf_greedy(bandit, num_rounds, xi=1, type_of_pull='single'):
    """Function that reproduces the steps involved in epsilon greedy
    algorithm"""

    for round in range(num_rounds):
        flip = random.random()
        if flip < calc_eps_n(bandit, xi):
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
            else:
                bandit.pull_arm(arm_var)
        else:
            # if random flip is greater than threshold, we exploit
            # pull the arm with max reward so far
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(bandit.max_reward_arm, prop_lis=
                [1 if i==bandit.max_reward_arm else 0 for i in range(bandit.num_arms)])
            else:
                bandit.pull_arm(bandit.max_reward_arm)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_rounds = 1000
    arm_means = [1,2,3,4]
    arm_vars = [1,1,1,1]
    epsilon_bandit = Bandit(name='epsilon_greedy',
                            arm_means=arm_means,
                            arm_vars=arm_vars)
    epsilon_inf_greedy(bandit=epsilon_bandit,
                       num_rounds=num_rounds)