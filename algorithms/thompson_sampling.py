from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, num_obs
import numpy as np
from math import sqrt


def bayesian_update(prior_params, x):
    """ m,k,v,s, are parameters of normal inverse chi squared distribution
    https: // www.cs.ubc.ca / ~murphyk / Papers / bayesGauss.pdf"""
    m, k, v, s = prior_params[0], prior_params[1], prior_params[2], \
                 prior_params[3]
    k1 = k + 1
    v1 = v + 1
    m1 = ((k*m)+x)/k1
    s1 = (1/v1)*(v*s + (k/(k+1))*((m-x)**2))
    return m1, k1, v1, s1
    
    
def thompson_sampling(bandit, num_rounds):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    
    # we initialize the distributions of arms to normal inverse chi squared
    # distibution. Each arm has params (m, k, v, s).
    # m is mean and s is variance
    
    num_arms = bandit.num_arms
    
    prior_params = [(0, 1, 1, 1) for i in range(num_arms)]
    
    for rnd in range(num_rounds):
        # we store all sampled values in this list
        sample = []
        # we sample each arm from the prior distribution
        for arm in range(num_arms):
            current_mean = prior_params[arm][0]
            current_var = prior_params[arm][3]
            sample.append(np.random.normal(current_mean, sqrt(current_var), 1))
        # the arm that has the highest value of draw is selected
        chosen_arm = np.argmax(sample)
        # That chosen arm is pulled to observe reward
        bandit.pull_arm(chosen_arm)
        reward = bandit.reward_tracker[-1]
        # posterior distribution is estimated using bayesian update
        prior_params[chosen_arm] = bayesian_update(prior_params[chosen_arm],
                                                   reward)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = 1000
    trt_dist_lis_th = trt_dist_list[:num_arms]
    thompson_bandit = Bandit(name='thompson_sampling',
                            num_arms=num_arms,
                            trt_dist_list=trt_dist_lis_th)
    thompson_sampling(thompson_bandit, num_rounds=num_rounds)