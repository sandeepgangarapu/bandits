from bandits.algorithms.explore_first import explore_first
from bandits.algorithms.epsilongreedy import epsilon_greedy
from bandits.algorithms.hte import always_explore
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.successive_elimination import successive_elimination
from bandits.distributions import trt_dist_list, num_obs
from bandits.bandit import Bandit
import matplotlib.pyplot as plt

def compare_bandits(bandit_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = range(1,num_obs+1)
    for b in bandit_list:
        ax.plot(x, b.regret, label=b.name)
    plt.legend()
    plt.savefig('comp.png')
    plt.show()
    plt.clf()
    
if __name__ == '__main__':
    num_arms = 4
    num_rounds = num_obs
    trt_dist_lis = trt_dist_list[:num_arms]
    always_explore_bandit = Bandit(always_explore, num_arms,
                                                  trt_dist_lis)
    always_explore_bandit = always_explore(always_explore_bandit,
                                           num_rounds, num_arms)
    explore_percentage = 10
    explore_first_bandit = explore_first(Bandit(explore_first, num_arms,
                                                trt_dist_lis),
                                         num_rounds, explore_percentage,
                                         num_arms)
    epsilon = 0.3
    epsilon_greedy_bandit = epsilon_greedy(epsilon, Bandit(epsilon_greedy,
                                                           num_arms,
                                                           trt_dist_lis),
                                           num_rounds, num_arms)
    successive_elimination_bandit = successive_elimination(Bandit(
        successive_elimination, num_arms,trt_dist_lis),
                                         num_rounds, num_arms)
    ucb_naive_bandit = ucb_naive(Bandit(ucb_naive, num_arms, trt_dist_lis),
                                 num_rounds,
                                 num_arms)
    bandit_list = [always_explore_bandit, explore_first_bandit,
                   epsilon_greedy_bandit, ucb_naive_bandit]
    compare_bandits(bandit_list)

