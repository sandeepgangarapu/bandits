from math import sqrt, log
import numpy as np
import random
from collections import Counter
from sklearn.metrics import mean_squared_error


num_obs = 100000

random.seed(8153003)
control = np.random.normal(loc=1, scale=1.0, size=num_obs)
trt1 = np.random.normal(loc=1.5, scale=1.0, size=num_obs)
trt2 = np.random.normal(loc=2.0, scale=1.0, size=num_obs)
trt3 = np.random.normal(loc=5.5, scale=2.0, size=num_obs)

trt_dist_list = [control, trt1, trt2, trt3]


# from contextual bandits
control = np.hstack([np.random.normal(loc=4, scale=1.0, size=num_obs),
                    np.random.normal(loc=3, scale=1.0, size=num_obs),
                    np.random.normal(loc=2, scale=1.0, size=num_obs),
                    np.random.normal(loc=1, scale=1.0, size=num_obs)])

trt1 = np.hstack([np.random.normal(loc=3, scale=1.0, size=num_obs),
                    np.random.normal(loc=2.8, scale=1.0, size=num_obs),
                    np.random.normal(loc=5, scale=1.0, size=num_obs),
                    np.random.normal(loc=6, scale=1.0, size=num_obs)])
hte_trt_dist_list = [control, trt1]


def create_distributions_vanilla(num_arms, return_mean_var=False):
    control_mean = 0
    variance = 1
    # this is not num_subjects
    size = 500
    # pick arms from uniform distribution between 0 and num_arms
    arm_means = np.random.uniform(0, num_arms/10, num_arms-1)
    # we create lis of lis for all distributions
    dist_list = []
    # add control list
    ctrl = np.random.normal(loc=control_mean, scale=variance, size=size)
    dist_list.append(ctrl)
    for i in arm_means:
        dis = np.random.normal(loc=i, scale=variance, size=size)
        dist_list.append(dis)
    if return_mean_var:
        arm_vars = [variance for i in range(num_arms)]
        arm_means = np.insert(arm_means, 0, control_mean, axis=0)
        return dist_list, arm_means, arm_vars
    return dist_list


def create_distributions_custom(arm_means, arm_vars, num_subjects):
    # this is not num_subjects
    size = num_subjects
    # pick arms from uniform distribution between 0 and num_arms
    # we create lis of lis for all distributions
    dist_list = []
    for i, j in zip(arm_means, arm_vars):
        dis = np.random.normal(loc=i, scale=sqrt(j), size=size)
        dist_list.append(dis)
    return dist_list


def ucb_value_naive(num_arms, num_rounds, arm_pull_tracker,
                    avg_reward_tracker, var_reward_tracker,
                    type_of_pull='single'):
    ucb = [0 for i in range(num_arms)]
    for arm in range(num_arms):
        conf_interval = sqrt((2*log(num_rounds))/(arm_pull_tracker[arm]))
        ucb[arm] = avg_reward_tracker[arm] + conf_interval
    chosen_arm = np.argmax(ucb)
    if type_of_pull == 'single':
        return chosen_arm
    winning_arm_lis = [chosen_arm]
    if type_of_pull == 'monte_carlo':
        num_pulls = 3000
        conf_interval_arms = np.sqrt((2 * log(num_rounds)) / np.array(arm_pull_tracker))
        for pull in range(num_pulls):
            sample = []
            for arm in range(num_arms):
                current_mean = avg_reward_tracker[arm]
                current_var = var_reward_tracker[arm]
                random_sample_mean = np.random.normal(current_mean, sqrt(current_var/arm_pull_tracker[arm]), 1)
                ucb_local = random_sample_mean[0] + conf_interval_arms[arm]
                sample.append(ucb_local)
            # the arm that has the highest value of draw is selected
            winning_arm_lis.append(np.argmax(sample))
        arm_counter = Counter(winning_arm_lis)
        prop_score_lis = []
        for arm in range(num_arms):
            prop_score_lis.append(float(arm_counter[arm] / len(winning_arm_lis)))
        return chosen_arm, prop_score_lis


def lcb_value_naive(num_arms, num_rounds, arm_pull_tracker,
                    avg_reward_tracker):
    lcb = [0 for i in range(num_arms)]
    for arm in range(num_arms):
        conf_interval = sqrt((2*log(num_rounds))/(arm_pull_tracker[arm]))
        lcb[arm] = avg_reward_tracker[arm] - conf_interval
    return lcb


def treatment_outcome_grouping(group, outcome, group_outcome=False,
                               all_arms=False, num_arms = None):
    """
    :param group: list of allocated groups e.g., [0,1,0,2,3,1]
    :param outcome: list of outcomes e.g., [1.5,2.1,3.4,5,9.34]
    :return: outcome list_of_lis
    """
    unique_arms = np.sort(np.unique(group))
    outcome_lis_of_lis = []
    for arm in unique_arms:
        outcome_lis_of_lis.append([outcome[j] for j in range(len(group)) if group[j] == arm])
        
    if group_outcome:
        return unique_arms, outcome_lis_of_lis
    elif all_arms:
        arms = list(range(num_arms))
        outcome_lis_of_lis = []
        for arm in arms:
            outcome_lis_of_lis.append(
                [outcome[j] for j in range(len(group)) if group[j] == arm])
        return arms, outcome_lis_of_lis
    else:
        return outcome_lis_of_lis


def thompson_arm_pull(mean_lis, var_lis, type_of_pull='single'):
    """
    :param mean_lis: mean list of all arms
    :param var_lis: variance list of all arms
    :param type_of_pull: either single pull or monte carlo pull
    :return: if single pull, gives out only winning arm
             if monte carlo pull, gives out winning arm and prop_scores
    """
    winning_arm = []
    # we store all sampled values in this list
    sample = []
    # we sample each arm from the prior distribution
    for arm in range(len(mean_lis)):
        current_mean = mean_lis[arm]
        current_var = var_lis[arm]
        sample.append(np.random.normal(current_mean, sqrt(current_var)))
    # the arm that has the highest value of draw is selected
    chosen_arm = np.argmax(sample)
    winning_arm.append(chosen_arm)
    if type_of_pull == 'single':
        return chosen_arm

    if type_of_pull == 'monte_carlo':
        num_pulls = 2000
        for pull in range(num_pulls):
            sample = []
            for arm in range(len(mean_lis)):
                current_mean = mean_lis[arm]
                current_var = var_lis[arm]
                sample.append(np.random.normal(current_mean, sqrt(current_var)))
            # the arm that has the highest value of draw is selected
            winning_arm.append(np.argmax(sample))
        arm_counter = Counter(winning_arm)
        prop_score_lis = []
        for arm in range(len(mean_lis)):
            prop_score_lis.append(float(arm_counter[arm]/len(winning_arm)))

        return chosen_arm, prop_score_lis


def bayesian_update(prior_params, x):
    """ m,k,v,s, are parameters of normal inverse chi squared distribution
    https: // www.cs.ubc.ca / ~murphyk / Papers / bayesGauss.pdf"""
    m, k, v, s = prior_params[0], prior_params[1], prior_params[2], \
                 prior_params[3]
    k1 = k + 1
    v1 = v + 1
    m1 = ((k * m) + x) / k1
    s1 = (1 / v1) * (v * s + (k / (k + 1)) * ((m - x) ** 2))
    return m1, k1, v1, s1


def bayesian_update_normal_inv_gamma(prior_params, lis_x):
    """ https://www.coursera.org/lecture/bayesian/the-normal-gamma-conjugate-family-ncApT"""
    m, n, s, d = prior_params[0], prior_params[1], prior_params[2], \
                 prior_params[3]
    n0 = len(lis_x)
    d0 = n0-1
    m0 = np.mean(lis_x)
    s0 = np.var(lis_x, ddof=1)
    # parameter update
    n1 = n+n0
    d1 = d+n0
    m1 = ((n * m) + (n0 * m0)) / n1
    s1 = (1 / d1) * ((s*d) + (s0*d0) + (((n0*n)/n1)*((m-m0)**2)))
    return m1, n1, s1, d1


def time_lis_of_lis(lis):
    """
    :param lis: list of anything [9,8,4,4,8,9,6]
    :return: list of lists = [[9],[9,8],[9,8,4],[9,8,4,4]......]
    """
    # Given outcome list, this func gives the outcomes so far in a list
    # Output is a list of list
    time_lis_of_lis = []
    for i in range(1, len(lis)):
        time_lis_of_lis.append(lis[:i])
    return time_lis_of_lis


def overall_stats(outcome):
    """
    :param outcome: list of outcomes as per allocation
    :return: dict of stats
    """
    outcome_time_lis_of_lis = time_lis_of_lis(outcome)
    avg = [np.mean(lis) for lis in outcome_time_lis_of_lis]
    var = [np.var(lis) for lis in outcome_time_lis_of_lis]
    return {'avg': avg, 'var': var}


def regret_outcome(group, outcome):
    """
    :param group: list of group numbers
    :param outcome: list of outcomes
    :return: regret and rmse of variance
    """

    regret = []
    for i in range(1, len(group) + 1):
        grp = group[:i]
        out = outcome[:i]
        arm, outcome_lis_of_lis = treatment_outcome_grouping(grp, out,
                                                             group_outcome=True)
        avg = [np.mean(lis) for lis in outcome_lis_of_lis]
        reg = i * np.max(avg) - sum(out)
        regret.append(reg)

    return regret


def rmse_outcome(group, outcome, true_mean, true_var):
    """
    :param group: list of group numbers
    :param outcome: list of outcomes
    :param true_mean: list of true means if arms
    :param true_var: list of tru variances of arms
    :return: mean rmse and variance rmse lists
    """
    mean_rmse_lis = []
    var_rmse_lis = []
    for i in range(1, len(group) + 1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], outcome[:i], group_outcome=True)
        mean_lis = [np.mean(lis) for lis in outcome_lis_of_lis]
        var_lis = [np.var(lis) for lis in outcome_lis_of_lis]
        mean_true = []
        var_true = []
        for arm in arms:
            mean_true.append(true_mean[arm])
            var_true.append(true_var[arm])
        mean_rmse = mean_squared_error(mean_true, mean_lis)
        var_rmse = mean_squared_error(var_true, var_lis)
        mean_rmse_lis.append(mean_rmse)
        var_rmse_lis.append(var_rmse)
    # rmse_plot(group, outcome, true_mean, true_var)
    return mean_rmse_lis, var_rmse_lis


def mse_outcome(group, outcome, true_mean, true_var):
    """
    :param group: list of group numbers
    :param outcome: list of outcomes
    :param true_mean: list of true means if arms
    :param true_var: list of tru variances of arms
    :return: mean rmse and variance rmse lists
    """
    mean_mse_lis = []
    var_mse_lis = []
    for i in range(1, len(group) + 1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], outcome[:i], group_outcome=True)
        mean_lis = [np.mean(lis) for lis in outcome_lis_of_lis]
        var_lis = [np.var(lis) for lis in outcome_lis_of_lis]
        mean_true = []
        var_true = []
        for arm in arms:
            mean_true.append(true_mean[arm])
            var_true.append(true_var[arm])
        mean_mse = mean_squared_error(mean_true, mean_lis)
        var_mse = mean_squared_error(var_true, var_lis)
        mean_mse_lis.append(mean_mse)
        var_mse_lis.append(var_mse)
    # rmse_plot(group, outcome, true_mean, true_var)
    return mean_mse_lis, var_mse_lis


def prop_mse(group, mean_est, true_est):
    """
    given group list and list of estimated means, the outcome will be mse at each point of time
    :param group: list of group numbers
    :param outcome: list of outcomes (in this case will be arm means)
    :param true_mean: list of true means if arms
    :param true_var: list of tru variances of arms
    :return: mean mse and variance mse lists
    """
    mean_mse_lis = []
    for i in range(1, len(group) + 1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], mean_est[:i], group_outcome=True)
        mean_lis = [lis[-1] for lis in outcome_lis_of_lis]
        mean_true = []
        for arm in arms:
            mean_true.append(true_est[arm])
        mean_mse = mean_squared_error(mean_true, mean_lis)
        mean_mse_lis.append(mean_mse)
    return mean_mse_lis


def rmse_plot(group, outcome, true_mean, true_var):
    mean_rmse_lis = []
    var_rmse_lis = []
    var_lis_lis = []
    for i in range(1, len(group) + 1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], outcome[:i], group_outcome=True)
        mean_lis = [np.mean(lis) for lis in outcome_lis_of_lis]
        var_lis = [np.var(lis) for lis in outcome_lis_of_lis]
        lis = []
        arms = np.array(arms)
        for a in range(10):
            if a in arms:
                lis.append(var_lis[np.where(arms == a)[0][0]])
            else:
                lis.append(0)

        var_lis_lis.append(lis)
        mean_true = []
        var_true = []
        for arm in arms:
            mean_true.append(true_mean[arm])
            var_true.append(true_var[arm])
        mean_rmse = sqrt(mean_squared_error(mean_true, mean_lis))
        var_rmse = sqrt(mean_squared_error(var_true, var_lis))
        mean_rmse_lis.append(mean_rmse)
        var_rmse_lis.append(var_rmse)

    return mean_rmse_lis, var_rmse_lis


def var_stats(group, outcome):
    var_lis_of_lis = []
    for i in range(1, len(group) + 1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], outcome[:i], group_outcome=False, all_arms=True, num_arms=10)
        var_lis = [np.var(lis) for lis in outcome_lis_of_lis]
        var_lis_of_lis.append(var_lis)
    return np.array(var_lis_of_lis).flatten()
