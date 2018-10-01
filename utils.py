from math import sqrt, log


def ucb_naive(num_arms, num_rounds, arm_pull_tracker, avg_reward_tracker):
    ucb = [0 for i in range(num_arms)]
    for arm in range(num_arms):
        conf_interval = sqrt((2*log(num_rounds))/(arm_pull_tracker[arm]))
        ucb[arm] = avg_reward_tracker[arm] + conf_interval
    return ucb


def lcb_naive(num_arms, num_rounds, arm_pull_tracker, avg_reward_tracker):
    lcb = [0 for i in range(num_arms)]
    for arm in range(num_arms):
        conf_interval = sqrt((2*log(num_rounds))/(arm_pull_tracker[arm]))
        lcb[arm] = avg_reward_tracker[arm] + conf_interval
    return lcb