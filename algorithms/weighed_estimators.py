import numpy as np
import pandas as pd
from math import sqrt

def get_final_mean(num_arms, arm_lis, mean_est_t):
    """
    This is a sub function that will give the final estimated means if a list of means of different arms at different
    points of time is given
    :param num_arms: total number of arms
    :param arm_lis: list of arm pulls
    :param mean_est_t: list of mean estimates of different arms at different t
    :return: final estimates of mean, one for each arm
    """
    # initializing empty array with num_arms elements
    mean_est_arm = [None] * num_arms
    num_updates = 0
    for arm_lis_ind in range(len(arm_lis) - 1, -1, -1):
        # we only update null array num_arm times, after that the ipw_mean_est_2 is full
        if num_updates >= num_arms:
            break
        else:
            arm_no = arm_lis[arm_lis_ind]
            # if the placeholder for the arm in empty, we replace, else we don't do anything
            if not mean_est_arm[arm_no]:
                mean_est_arm[arm_no] = mean_est_t[arm_lis_ind]
                num_updates = num_updates + 1
    return mean_est_arm


def weighed_estimators(type, arm_lis, reward_lis, weight_lis, type_of_eval_weight='constant_allocation',
                       weight_lis_of_lis=None, final_means=False, num_arms=None):
    """
    This is a function for inverse propensity score weighed estimator
    :param type: type of weighed estimator (ex: "ipw")
    :param arm_lis: list of arm pulls
    :param reward_lis: list of rewards
    :param weight_lis: list of weights
    :param final_means: BOOL - whether the final estimated means of each arm to be returned
    :return: Either list of estimated arm means at that time or final means
    """
    weight_lis = np.array(weight_lis)
    reward_lis = np.array(reward_lis)
    if not num_arms:
        num_arms = max(arm_lis) + 1
    # binary indicator of whether an arm is pulled at time t for all arms
    ind_arm = []
    inv_prop = 1/weight_lis
    for arm in range(num_arms):
        ind_arm.append(np.array([1 if j == arm else 0 for j in arm_lis]))
    weighed_reward_lis = reward_lis / weight_lis

    if type == "ipw":
        ipw_mean_est = []
        for i in range(len(arm_lis)):
            current_arm = arm_lis[i]
            weighed_reward = weighed_reward_lis[:i+1]
            ind_array = ind_arm[current_arm][:i+1]
            ipw_est = np.sum(weighed_reward*ind_array)/(i+1)
            ipw_mean_est.append(ipw_est)
        if final_means:
            ipw_mean_est = get_final_mean(num_arms, arm_lis, ipw_mean_est)
        return ipw_mean_est

    if type == "aipw":
        aipw_mean_est = []
        # this will initialize arm means of all arms to 0
        mean_snapshot = {key: [0] for key in range(num_arms)}
        for i in range(len(arm_lis)):
            current_arm = arm_lis[i]
            weighed_reward = weighed_reward_lis[:i+1]
            ind_array = ind_arm[current_arm][:i+1]
            inv_prop_array = inv_prop[:i+1]
            mean_array = np.array(mean_snapshot[current_arm])
            aipw_est = np.sum((weighed_reward*ind_array) + mean_array - (mean_array*ind_array*inv_prop_array))/(i+1)
            aipw_mean_est.append(aipw_est)
            for j in mean_snapshot.keys():
                if current_arm == j:
                    new_mean = ((mean_snapshot[j][-1]*len(mean_snapshot[j])) + reward_lis[i])/(len(mean_snapshot[j])+1)
                    mean_snapshot[j].append(new_mean)
                else:
                    mean_snapshot[j].append(mean_snapshot[j][-1])
        if final_means:
            aipw_mean_est = get_final_mean(num_arms, arm_lis, aipw_mean_est)
        return aipw_mean_est

    if type == "eval_aipw":
        global eval_weights
        if type_of_eval_weight == 'uniform':
            eval_weights = 1 / (np.array(list(range(1, len(arm_lis) + 1))))
        if type_of_eval_weight == 'poly_decay':
            alpha = 0.5
            eval_weights = 1 / (np.array(list(range(1, len(arm_lis) + 1))) ** alpha)
        if type_of_eval_weight == 'propensity_score':
            eval_weights = weight_lis
        if type_of_eval_weight == 'variance_stabilizing':
            eval_weights = []
            for j in range(len(arm_lis)):
                eval_wt = np.array(weight_lis[j]) / np.sum([lis[arm_lis[j]] for lis in weight_lis_of_lis])
                eval_weights.append(eval_wt)
            eval_weights = np.array(eval_weights)
        if type_of_eval_weight == 'constant_allocation':
            eval_weights = np.sqrt(weight_lis/len(arm_lis))
        eval_aipw_mean_est = []
        # this will initialize arm means of all arms to 0
        mean_snapshot = {key: [0] for key in np.unique(arm_lis)}
        for i in range(len(arm_lis)):
            current_arm = arm_lis[i]
            weighed_reward = weighed_reward_lis[:i+1]
            ind_array = ind_arm[current_arm][:i+1]
            inv_prop_array = inv_prop[:i+1]
            eval_array = eval_weights[:i+1]
            mean_array = np.array(mean_snapshot[current_arm])
            aipw_term = (weighed_reward * ind_array) + mean_array - (mean_array * ind_array * inv_prop_array)
            eval_aipw_est = np.sum(aipw_term * eval_array) / np.sum(eval_array)
            eval_aipw_mean_est.append(eval_aipw_est)
            for j in mean_snapshot.keys():
                if current_arm == j:
                    new_mean = ((mean_snapshot[j][-1] * len(mean_snapshot[j])) + reward_lis[i]) / (
                            len(mean_snapshot[j]) + 1)
                    mean_snapshot[j].append(new_mean)
                else:
                    mean_snapshot[j].append(mean_snapshot[j][-1])
        if final_means:
            eval_aipw_mean_est = get_final_mean(num_arms, arm_lis, eval_aipw_mean_est)
        return eval_aipw_mean_est


if __name__ == '__main__':
    dt_lis = []
    for j in range(100):
        print(j)
        means = np.random.uniform(0,1,4)
        vars = np.random.uniform(0,1,4)
        weights = np.random.uniform(0, 1, 4)
        weight_lis_true = weights / np.sum(weights)
        arm_lis = np.random.choice([0, 1, 2, 3], 1000, p=weight_lis_true)
        weight_lis = [weight_lis_true[k] for k in list(arm_lis)]
        for i in range(100):
            reward_lis = [np.random.normal(means[i], sqrt(vars[i])) for i in arm_lis]
            for alg in ['ipw', 'aipw', 'eval_aipw']:
                a = weighed_estimators(type=alg, arm_lis=arm_lis,
                                       reward_lis=reward_lis,
                                       weight_lis=weight_lis,
                                       final_means=True,
                                       num_arms=4)
                dt_sub = pd.DataFrame({'ite': np.repeat([i], 4),
                                       'main_ite': np.repeat([j], 4),
                                       'alg': np.repeat([alg], 4),
                                       'grp': [0, 1, 2, 3],
                                       'true_mean': means,
                                       'weight_lis': weight_lis_true,
                                       'mn': a})
                dt_lis.append(dt_sub)

    dt_final = pd.concat(dt_lis)
    dt_final.to_csv("test_weighed.csv", index=False)
