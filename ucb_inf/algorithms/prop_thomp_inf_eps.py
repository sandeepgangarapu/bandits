import numpy as np
from bandits.utils import thompson_arm_pull


def calc_eps_n(bandit, chi):
    sum_of_st_err = (np.array(bandit.var_est_tracker) / np.array(
        bandit.arm_pull_tracker)).sum()
    eta = sum_of_st_err / (bandit.num_arms * chi)
    eps_n = eta / (1 + eta)
    return eps_n


def prop_thomp_inf_eps(bandit, num_subjects, chi=1):
    print("---------------Running Thomp INF EPS ---------------")

    # # allocate one subject to each arm (We can remove this later rules)
    # for ite in range(2):
    #     for arm in range(bandit.num_arms):
    #         bandit.pull_arm(arm, propensity=1/bandit.num_arms)

    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects):
        # perc_ab of the time, we do variance based, otherwise UCB
        if np.random.uniform(0, 1) < calc_eps_n(bandit, chi):
            # variance base, we pick arm proportional to the variance of
            # mean estimate, i.e. s^2/n
            var_of_mean_est = \
                np.array(bandit.var_est_tracker) / np.array(
                    bandit.arm_pull_tracker)
            # normalizing std.err square for calculating probabilities
            norm_prob = [v / sum(var_of_mean_est) for v in var_of_mean_est]
            arm_var = np.random.choice(list(range(bandit.num_arms)),
                                       p=norm_prob)
            bandit.pull_arm(arm_var, propensity=norm_prob[arm_var])
        else:
            chosen_arm, prop_lis = thompson_arm_pull(m=bandit.avg_reward_tracker, s=bandit.var_est_tracker, type_of_pull='monte_carlo')
            bandit.pull_arm(chosen_arm, prop_lis[chosen_arm])
    # we return all of bandit instance as we cab use this to calculate both IPW and AIPW
    return bandit