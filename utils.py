from math import sqrt, log
import numpy as np
import random

num_obs = 10000

random.seed(8153003)
control = np.random.normal(loc=1, scale=1.0, size=num_obs)
trt1 = np.random.normal(loc=1.5, scale=1.0, size=num_obs)
trt2 = np.random.normal(loc=2.0, scale=1.0, size=num_obs)
trt3 = np.random.normal(loc=2.5, scale=2.0, size=num_obs)

trt_dist_list = [control, trt1]


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


def create_distributions_vanilla(num_arms):
    control_mean = 0
    variance = 1
    # this is not num_subjects
    size = 100000
    # pick arms from uniform distribution between 0 and num_arms
    arm_means = np.random.uniform(0, num_arms, num_arms-1)
    # we create lis of lis for all distributions
    dist_list = []
    # add control list
    ctrl = np.random.normal(loc=control_mean, scale=variance, size=size)
    dist_list.append(ctrl)
    for i in arm_means:
        dis = np.random.normal(loc=i, scale=variance, size=size)
        dist_list.append(dis)
    return dist_list


def create_distributions_custom(arm_means, arms_vars):
    # this is not num_subjects
    size = 100000
    # pick arms from uniform distribution between 0 and num_arms
    # we create lis of lis for all distributions
    dist_list = []
    for i, j in zip(arm_means, arms_vars):
        dis = np.random.normal(loc=i, scale=j, size=size)
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