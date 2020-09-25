import numpy as np
import pandas as pd
from math import sqrt

def weighed_estimators(type, arm_lis, reward_lis, weight_lis,
                       weight_lis_of_lis=None, num_arms=None):
    """
    This is a function for inverse propensity score weighed estimator
    :param type: type of weighed estimator (ex: "ipw")
    :param arm_lis: list of arm pulls
    :param reward_lis: list of rewards
    :param weight_lis: list of weights
    :param final_means: BOOL - whether the final estimated means of each arm to be returned
    :return: Either list of estimated arm means at that time or final means
    """
    horizon = len(arm_lis)
    weight_lis = np.array(weight_lis)
    reward_lis = np.array(reward_lis)
    inv_prop = 1 / weight_lis
    weighed_reward_lis = reward_lis * inv_prop
    if not num_arms:
        num_arms = max(arm_lis) + 1
    # binary indicator of whether an arm is pulled at time t for all arms
    ind_arm = []
    for arm in range(num_arms):
        ind_arm.append(np.array([1 if j == arm else 0 for j in arm_lis]))

    if type == "ipw":
        ipw_mean_est = []
        for i in range(num_arms):
            ipw_est = np.sum(weighed_reward_lis * ind_arm[i]) / (horizon)
            ipw_mean_est.append(ipw_est)
        return ipw_mean_est

    if type == "aipw":
        aipw_mean_est = []
        for i in range(num_arms):
            # reward of a particular arm
            rew_arm = reward_lis * ind_arm[i]
            # now we find sample mean at any point of time
            rew_arm_cum_sum = np.cumsum(rew_arm)
            denom = np.arange(1, horizon+1) * ind_arm[i]
            placeholder = 1
            for m in range(len(denom)):
                if denom[m]==0:
                    denom[m] = placeholder
                else:
                    placeholder = denom[m]
            mean_snapshot = rew_arm_cum_sum/denom
            # we now insert 0 at the start of this array so that we get the sample mean until that time and not
            # including that time. This is as per eq 5 of athey
            mean_snapshot_final = np.insert(mean_snapshot, 0, 0)[:-1]
            aipw_est = np.sum(
                (weighed_reward_lis * ind_arm[i]) + mean_snapshot_final - (mean_snapshot_final * ind_arm[i] * inv_prop)) / (horizon)
            aipw_mean_est.append(aipw_est)
        return aipw_mean_est

    if type == "eval_aipw":
        eval_aipw_mean_est = []
        for i in range(num_arms):
            # reward of a particular arm
            rew_arm = reward_lis * ind_arm[i]
            # now we find sample mean at any point of time
            rew_arm_cum_sum = np.cumsum(rew_arm)
            denom = np.arange(1, horizon + 1)
            placeholder = 1
            for m in range(len(denom)):
                if denom[m] == 0:
                    denom[m] = placeholder
                else:
                    placeholder = denom[m]
            mean_snapshot = rew_arm_cum_sum / denom
            # we now insert 0 at the start of this array so that we get the sample mean until that time and not
            # including that time. This is as per eq 5 of athey
            mean_snapshot_final = np.insert(mean_snapshot, 0, 0)[:-1]
            aipw_array = (weighed_reward_lis * ind_arm[i]) + mean_snapshot_final - (
                            mean_snapshot_final * ind_arm[i] * inv_prop)
            eval_array = np.sqrt((np.array(weight_lis_of_lis)[:, i]/horizon))
            eval_aipw_est = np.sum(aipw_array * eval_array) / np.sum(eval_array)
            eval_aipw_mean_est.append(eval_aipw_est)
        return eval_aipw_mean_est

    if type == "eval_aipw_var":
        eval_aipw_var_est = []
        for i in range(num_arms):
            # reward of a particular arm
            rew_arm = reward_lis * ind_arm[i]
            # now we find sample mean at any point of time
            rew_arm_cum_sum = np.cumsum(rew_arm)
            denom = np.arange(1, horizon + 1)
            placeholder = 1
            for m in range(len(denom)):
                if denom[m] == 0:
                    denom[m] = placeholder
                else:
                    placeholder = denom[m]
            mean_snapshot = rew_arm_cum_sum / denom
            # we now insert 0 at the start of this array so that we get the sample mean until that time and not
            # including that time. This is as per eq 5 of athey
            mean_snapshot_final = np.insert(mean_snapshot, 0, 0)[:-1]
            aipw_array = (weighed_reward_lis * ind_arm[i]) + mean_snapshot_final - (
                            mean_snapshot_final * ind_arm[i] * inv_prop)
            eval_array = np.sqrt((np.array(weight_lis_of_lis)[:, i]/horizon))
            eval_aipw_mean_est = np.sum(aipw_array * eval_array) / np.sum(eval_array)
            var_est = np.sum(np.square(eval_array)*np.square(aipw_array - eval_aipw_mean_est)) / ((np.sum(eval_array))**2)
            if var_est<0:
                print("hello")
            eval_aipw_var_est.append(var_est)
        return eval_aipw_var_est

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
