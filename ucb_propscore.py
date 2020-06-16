from bandits.algorithms.ucb import ucb
from bandits.algorithms.ucb_inf_eps import ucb_inf_eps
import numpy as np
from collections import Counter
from math import sqrt
from bandits.bandit import Bandit
from scipy.spatial.distance import cosine
from sklearn.metrics.regression import mean_squared_error
from bandits.algorithms.weighed_estimators.weighed_estimators import ipw, aipw
import pandas as pd


true_means = [0.25, 1.82, 1.48, 2.25, 2]
true_vars = [2.84,  1.97, 2.62, 1, 2.06]

# true_means = [0.25, 2.25, 2]
# true_vars = [2.84, 1, 2.06]
#

# true_means = [1,2,3]
# true_vars = [1,1,1]


num_ite = 50
num_sub_ite = 1000000
horizon = 18
num_arms = len(true_means)
outcome_lis_of_lis = []
df_lis = []
for arm in range(num_arms):
    outcome_lis_of_lis.append(np.random.normal(loc=true_means[arm],
                                               scale=sqrt(true_vars[arm]),
                                              size=100000))
group_lis_of_lis = []

for j in range(num_sub_ite):
    ucb_bandit = Bandit(name='ucb',
                        num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)

    ucb(ucb_bandit, horizon)
    group_lis_of_lis.append(ucb_bandit.arm_tracker)

    
for ite in range(num_ite):
    print("ITE NO:", ite)
    
    ucb_bandit = Bandit(name='ucb',
                        num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)
    
    ucb(ucb_bandit, horizon, type_of_pull='monte_carlo')
    seed_list = ucb_bandit.arm_tracker.copy()
    rew_list = ucb_bandit.reward_tracker.copy()
    prop_gauss = ucb_bandit.propensity_tracker.copy()

    prop_lis_seq = []
    for i in range(len(seed_list)):
        group_lis_of_lis2 = group_lis_of_lis.copy()
        seed_group = seed_list[i]
        groups_at_i = []
        pop_list = []
        for j in range(len(group_lis_of_lis2)):
            if group_lis_of_lis2[j][:i] != seed_list[:i]:
                pop_list.append(j)
        # print(pop_list)
        for index in sorted(pop_list, reverse=True):
            del group_lis_of_lis2[index]
        for j in range(len(group_lis_of_lis2)):
            groups_at_i.append(group_lis_of_lis2[j][i])
        print(len(group_lis_of_lis2))
        arm_counter = Counter(groups_at_i)
        propensity = arm_counter[seed_group]/len(groups_at_i)
        prop_lis_seq.append(propensity)

    
    prop_lis_num = []
    for i in range(len(seed_list)):
        group_lis_of_lis3 = group_lis_of_lis.copy()
        seed_group = seed_list[i]
        groups_at_i = []
        pop_list = []
        for j in range(len(group_lis_of_lis3)):
            if Counter(group_lis_of_lis3[j][:i]) != Counter(seed_list[:i]):
                pop_list.append(j)
        # print(pop_list)
        for index in sorted(pop_list, reverse=True):
            del group_lis_of_lis3[index]
        for j in range(len(group_lis_of_lis3)):
            groups_at_i.append(group_lis_of_lis3[j][i])
        print(len(group_lis_of_lis3))
        arm_counter = Counter(groups_at_i)
        propensity = arm_counter[seed_group]/len(groups_at_i)
        prop_lis_num.append(propensity)
    
    ipw_gauss = ipw(seed_list, rew_list, prop_gauss)
    aipw_gauss = aipw(seed_list, rew_list, prop_gauss)
    ipw_seq = ipw(seed_list, rew_list, prop_lis_seq)
    aipw_seq = aipw(seed_list, rew_list, prop_lis_seq)
    ipw_num = ipw(seed_list, rew_list, prop_lis_num)
    aipw_num = aipw(seed_list, rew_list, prop_lis_num)
    print(prop_lis_seq)
    print(prop_gauss)
    print(prop_lis_num)
    dict_df = {'group': seed_list,
               'outcome': rew_list,
               'ite': np.repeat(ite, len(seed_list)),
               'ipw_gauss': ipw_gauss,
               'aipw_gauss': aipw_gauss,
               'ipw_seq': ipw_seq,
               'aipw_seq': aipw_seq,
               'ipw_num': ipw_num,
               'aipw_num': aipw_num,
               }
    df = pd.DataFrame(dict_df)
    df_lis.append(df)
    final_output = pd.concat(df_lis)
    final_output.to_csv('analysis/output/sim_ucb_seq_num_50_18'
                                        '.csv', index=False)
