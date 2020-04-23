from math import sqrt, log
import numpy as np
import random
from collections import Counter

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
                    avg_reward_tracker):
    ucb = [0 for i in range(num_arms)]
    for arm in range(num_arms):
        conf_interval = sqrt((2*log(num_rounds))/(arm_pull_tracker[arm]))
        ucb[arm] = avg_reward_tracker[arm] + conf_interval
    return ucb


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


def thompson_arm_pull(m, s, type_of_pull='single'):
    """
    :param m: mean list of all arms
    :param s: variance list of all arms
    :param type_of_pull: either single pull or monte carlo pull
    :return: if single pull, gives out only winning arm
             if monte carlo pull, gives out winning arm and prop_scores
    """
    winning_arm = []
    # we store all sampled values in this list
    sample = []
    # we sample each arm from the prior distribution
    for arm in range(len(m)):
        current_mean = m[arm]
        current_var = s[arm]
        sample.append(np.random.normal(current_mean, sqrt(current_var), 1))
    # the arm that has the highest value of draw is selected
    chosen_arm = np.argmax(sample)
    winning_arm.append(chosen_arm)
    if type_of_pull == 'single':
        return chosen_arm

    if type_of_pull == 'monte_carlo':
        num_pulls = 200
        for pull in range(num_pulls):
            sample = []
            for arm in range(len(m)):
                current_mean = m[arm]
                current_var = s[arm]
                sample.append(np.random.normal(current_mean, sqrt(current_var), 1))
            # the arm that has the highest value of draw is selected
            winning_arm.append(np.argmax(sample))
        arm_counter = Counter(winning_arm)
        prop_score_lis = []
        for arm in range(len(m)):
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


