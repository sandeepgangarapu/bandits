from bandits.algorithms.ucb import ucb
from bandits.algorithms.ucb_inf_eps import ucb_inf_eps
import numpy as np
from collections import Counter
from math import sqrt
from bandits.bandit import Bandit
from scipy.spatial.distance import cosine
from sklearn.metrics.regression import mean_squared_error
from bandits.algorithms.weighed_estimators.weighed_estimators import ipw, aipw
# true_means = [0.25, 2.82, 1.48, 4.95, 3]
# true_vars = [3.84,  1.97, 2.62, 1, 4.06]

true_means = [0.25, 2.25, 2]
true_vars = [2.84, 1, 2.06]


num_ite = 40
num_sub_ite = 50000
horizon = 12
num_arms = len(true_means)
outcome_lis_of_lis = []
group_lis_of_lis = []
for arm in range(num_arms):
    outcome_lis_of_lis.append(np.random.normal(loc=true_means[arm],
                                               scale=sqrt(true_vars[arm]),
                                              size=100000))

for ite in range(num_ite):
    
    ucb_bandit = Bandit(name='ucb',
                        num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)
    
    ucb_inf_eps(ucb_bandit, horizon, type_of_pull='monte_carlo')
    seed_list = ucb_bandit.arm_tracker
    prop_lis_gauss = ucb_bandit.propensity_tracker
    
    for j in range(num_sub_ite):
        ucb_bandit = Bandit(name='ucb',
                            num_arms=num_arms,
                            trt_dist_list=outcome_lis_of_lis)
    
        ucb_inf_eps(ucb_bandit, horizon)
        group_lis_of_lis.append(ucb_bandit.arm_tracker)
    
    
    # output = sim.run_simulation()
    # group_lis_of_lis = output.groupby('ite')['group'].apply(list).reset_index(name='group_lis')
    # group_list = group_lis_of_lis['group_lis'].to_list()
    #seed_list = group_lis_of_lis.pop(0)
    
    prop_lis_seq = []
    for i in range(len(seed_list)):
        seed_group = seed_list[i]
        sub_group = []
        pop_list = []
        for j in range(len(group_lis_of_lis)):
            sub_group.append(group_lis_of_lis[j][i])
            if group_lis_of_lis[j][i] != seed_group:
                pop_list.append(j)
        # print(pop_list)
        for index in sorted(pop_list, reverse=True):
            del group_lis_of_lis[index]
        print(len(group_lis_of_lis))
        arm_counter = Counter(sub_group)
        propensity = arm_counter[seed_group]/len(sub_group)
        prop_lis_seq.append(propensity)
    
    
    prop_lis_num = []
    for i in range(len(seed_list)):
        seed_group = seed_list[i]
        sub_group = []
        pop_list = []
        for j in range(len(group_lis_of_lis)):
            sub_group.append(group_lis_of_lis[j][i])
            if Counter(group_lis_of_lis[j][:i+1]) != Counter(seed_list[:i+1]):
                pop_list.append(j)
        # print(pop_list)
        for index in sorted(pop_list, reverse=True):
            del group_lis_of_lis[index]
        print(len(group_lis_of_lis))
        arm_counter = Counter(sub_group)
        propensity = arm_counter[seed_group]/len(sub_group)
        prop_lis_num.append(propensity)
    
    
    a = 1-cosine(prop_lis_seq, prop_lis_gauss)
    b = 1-cosine(prop_lis_seq, prop_lis_num)
    c = 1-cosine(prop_lis_num, prop_lis_gauss)
    
    
    print(prop_lis_gauss)
    # print(prop_lis_seq)
    print(prop_lis_num)
    print(c)
    d = mean_squared_error(prop_lis_seq, prop_lis_gauss)
    e = mean_squared_error(prop_lis_seq, prop_lis_num)
    f = mean_squared_error(prop_lis_num, prop_lis_gauss)
    
    print(f)
    
    
