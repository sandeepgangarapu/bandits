import numpy as np
from bandits.utils import treatment_outcome_grouping

def ipw(arm_lis, reward_lis, weight_lis):
    ipw_mean_est = []
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        weighed_reward = np.array(reward_lis[:i+1]) / np.array(weight_lis[:i+1])
        ind_array = np.array([1 if j == current_arm else 0 for j in arm_lis[:i+1]])
        ipw_est = np.sum(weighed_reward*ind_array)/(i+1)
        ipw_mean_est.append(ipw_est)
    return ipw_mean_est


def aipw(arm_lis, reward_lis, weight_lis):
    aipw_mean_est = []
    # this will initialize arm means of all arms to 0
    mean_snapshot = [[0] for i in range(len(np.unique(arm_lis)))]
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        weighed_reward = np.array(reward_lis[:i+1]) / np.array(weight_lis[:i+1])
        ind_array = np.array([1 if j == current_arm else 0 for j in arm_lis[:i+1]])
        mean_array = np.array(mean_snapshot[current_arm])
        aipw_est = np.sum((weighed_reward*ind_array) + mean_array - (mean_array*ind_array))/(i+1)
        aipw_mean_est.append(aipw_est)
        for j in range(len(mean_snapshot)):
            if current_arm == j:
                new_mean = ((mean_snapshot[j][-1]*len(mean_snapshot[j])) + reward_lis[i])/(len(mean_snapshot[j])+1)
                mean_snapshot[j].append(new_mean)
            else:
                mean_snapshot[j].append(mean_snapshot[j][-1])
    return aipw_mean_est


a = aipw([0,1,1,0,1,1], [1,1,1,1,1,1], [0.5, 0.5, 0.5,0.5,0.5,0.5])
print(a)