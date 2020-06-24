import numpy as np

def ipw(arm_lis, reward_lis, weight_lis, final_means = False):
    ipw_mean_est = []
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        weighed_reward = np.array(reward_lis[:i+1]) / np.array(weight_lis[:i+1])
        ind_array = np.array([1 if j == current_arm else 0 for j in arm_lis[:i+1]])
        ipw_est = np.sum(weighed_reward*ind_array)/(i+1)
        ipw_mean_est.append(ipw_est)
    if final_means:
        ipw_mean_est_2 = []
        num_arms = max(arm_lis)+1
        for arm1 in range(num_arms):
            for arm_lis_ind in range(len(arm_lis)-1, -1, -1):
                if arm1 == arm_lis[arm_lis_ind]:
                    ipw_mean_est_2.append(ipw_mean_est[arm_lis_ind])
                    break
        ipw_mean_est = ipw_mean_est_2.copy()
    return ipw_mean_est


def aipw(arm_lis, reward_lis, weight_lis, final_means = False):
    aipw_mean_est = []
    # this will initialize arm means of all arms to 0
    mean_snapshot = {key:[0] for key in np.unique(arm_lis)}
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        inv_prop = 1 / np.array(weight_lis[:i+1])
        weighed_reward = np.array(reward_lis[:i+1]) * inv_prop
        ind_array = np.array([1 if k == current_arm else 0 for k in arm_lis[:i+1]])
        mean_array = np.array(mean_snapshot[current_arm])
        aipw_est = np.sum((weighed_reward*ind_array) + mean_array - (mean_array*ind_array*inv_prop))/(i+1)
        aipw_mean_est.append(aipw_est)
        for j in mean_snapshot.keys():
            if current_arm == j:
                new_mean = ((mean_snapshot[j][-1]*len(mean_snapshot[j])) + reward_lis[i])/(len(mean_snapshot[j])+1)
                mean_snapshot[j].append(new_mean)
            else:
                mean_snapshot[j].append(mean_snapshot[j][-1])
    if final_means:
        aipw_mean_est_2 = []
        num_arms = max(arm_lis)+1
        for arm1 in range(num_arms):
            for arm_lis_ind in range(len(arm_lis) - 1, -1, -1):
                if arm1 == arm_lis[arm_lis_ind]:
                    aipw_mean_est_2.append(aipw_mean_est[arm_lis_ind])
                    break
        aipw_mean_est = aipw_mean_est_2.copy()
    return aipw_mean_est


def eval_aipw(arm_lis, reward_lis, weight_lis, type_of_weight='poly_decay', weight_lis_of_lis=None):
    if type_of_weight == 'uniform':
        eval_weights = 1/(np.array(list(range(1, len(arm_lis)+1))))
    if type_of_weight == 'poly_decay':
        alpha = 0.5
        eval_weights = 1/(np.array(list(range(1, len(arm_lis)+1)))**alpha)
    if type_of_weight == 'propensity_score':
        eval_weights = np.array(weight_lis)
    if type_of_weight == 'variance_stabilizing':
        eval_weights = []
        for j in range(len(arm_lis)):
            eval_wt = np.array(weight_lis[j])/np.sum([lis[arm_lis[j]] for lis in weight_lis_of_lis])
            eval_weights.append(eval_wt)
        eval_weights = np.array(eval_weights)

    eval_aipw_mean_est = []
    # this will initialize arm means of all arms to 0
    mean_snapshot = {key: [0] for key in np.unique(arm_lis)}
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        inv_prop = 1 / np.array(weight_lis[:i+1])
        weighed_reward = np.array(reward_lis[:i+1]) * inv_prop
        ind_array = np.array([1 if k == current_arm else 0 for k in arm_lis[:i + 1]])
        eval_array = eval_weights[:i+1]
        mean_array = np.array(mean_snapshot[current_arm])
        aipw_term = (weighed_reward * ind_array) + mean_array - (mean_array * ind_array * inv_prop)
        eval_aipw_est = np.sum(aipw_term*eval_array)/np.sum(eval_array)
        eval_aipw_mean_est.append(eval_aipw_est)
        for j in mean_snapshot.keys():
            if current_arm == j:
                new_mean = ((mean_snapshot[j][-1] * len(mean_snapshot[j])) + reward_lis[i]) / (
                            len(mean_snapshot[j]) + 1)
                mean_snapshot[j].append(new_mean)
            else:
                mean_snapshot[j].append(mean_snapshot[j][-1])
    return eval_aipw_mean_est


def athey_aipw(arm_lis, reward_lis, weight_lis, weight_lis_of_lis=None):

    eval_aipw_mean_est = []
    # this will initialize arm means of all arms to 0
    mean_snapshot = {key: [0] for key in np.unique(arm_lis)}
    for i in range(len(arm_lis)):
        current_arm = arm_lis[i]
        inv_prop = 1 / np.array(weight_lis[:i+1])
        weighed_reward = np.array(reward_lis[:i+1]) * inv_prop
        ind_array = np.array([1 if k == current_arm else 0 for k in arm_lis[:i + 1]])
        eval_array = eval_weights[:i+1]
        mean_array = np.array(mean_snapshot[current_arm])
        aipw_term = (weighed_reward * ind_array) + mean_array - (mean_array * ind_array * inv_prop)
        eval_aipw_est = np.sum(aipw_term*eval_array)/np.sum(eval_array)
        eval_aipw_mean_est.append(eval_aipw_est)
        for j in mean_snapshot.keys():
            if current_arm == j:
                new_mean = ((mean_snapshot[j][-1] * len(mean_snapshot[j])) + reward_lis[i]) / (
                            len(mean_snapshot[j]) + 1)
                mean_snapshot[j].append(new_mean)
            else:
                mean_snapshot[j].append(mean_snapshot[j][-1])
    return eval_aipw_mean_est




a = aipw([1,0,1,0],
         [1,1,2,2],
         [0.5, 0.5, 0.5, 0.5], final_means=False)

print(a)
