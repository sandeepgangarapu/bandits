from bandits.algorithms.explore_first import explore_first
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.algorithms.hte import always_explore
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.successive_elimination import successive_elimination
from bandits.distributions import trt_dist_list, num_obs
from bandits.bandit import Bandit
import matplotlib.pyplot as plt
from plotnine import *
import pandas as pd
from bandits.algorithms.linucb import linucb

def compare_bandits(bandit_list, num_rounds):
    df = pd.DataFrame([i for i in range(1, num_rounds+1)], columns=['x'])
    for b in bandit_list:
        df[b.name] = b.regret
    p = (ggplot(df)+ geom_line(aes('x', 'always_explore')))
    p.save('test.png')
    return df

    
    
if __name__ == '__main__':
    num_arms = 2
    num_rounds = 10000
    trt_dist_lis = trt_dist_list[:num_arms]
    always_explore_bandit = always_explore(Bandit("always_explore", num_arms,
                                                  trt_dist_lis),
                                           num_rounds, num_arms)
    explore_percentage = 10
    explore_first_bandit = explore_first(Bandit("explore_first", num_arms,
                                                trt_dist_lis),
                                         num_rounds, explore_percentage,
                                         num_arms)
    epsilon = 0.3
    epsilon_greedy_bandit = epsilon_greedy(epsilon, Bandit("epsilon_greedy",
                                                           num_arms,
                                                           trt_dist_lis),
                                           num_rounds, num_arms)
    print(epsilon_greedy_bandit.name)
    successive_elimination_bandit = successive_elimination(Bandit(
        "successive_elimination", num_arms,trt_dist_lis),
                                         num_rounds, num_arms)
    ucb_naive_bandit = ucb_naive(Bandit("ucb_naive", num_arms, trt_dist_lis),
                                 num_rounds,
                                 num_arms)
    bandit_list = [always_explore_bandit, explore_first_bandit,
                   epsilon_greedy_bandit, ucb_naive_bandit]
    data = compare_bandits(bandit_list, num_rounds)
    data['lin_ucb'] = linucb()
    print(data.head())
    