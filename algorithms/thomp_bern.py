from bandit import Bandit
from utils import thompson_arm_pull_bern
from math import sqrt


def thompson_sampling_bern(bandit, num_rounds, type_of_pull='single', cap_prop=False):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    print("---------------Running Thompson Sampling Bern---------------")

    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    num_initial_pulls = 2
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)

    # beta bernoulli updation process
    prior_params = [[1, 1] for i in range(num_arms)]

    for rnd in range(int(num_rounds - (num_initial_pulls * num_arms))):

        if type_of_pull == 'monte_carlo':
            chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
            # this is to cap probability of allocation as per Athey's algorthm
            if cap_prop:
                cap = 0.1 / sqrt(rnd + 1)
                chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull, cap_prop=cap)
            bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
        else:
            chosen_arm = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
            bandit.pull_arm(chosen_arm)
        x = bandit.reward_tracker[-1]
        # We calculate the posterior parameters of the beta distribution
        prior_params[chosen_arm][0] += x
        prior_params[chosen_arm][1] += 1-x
    return bandit


if __name__ == '__main__':
    # Define bandit
    for i in range(10):
        num_rounds = 1000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.1, 0.2, 0.3],
                                 dist_type='Bernoulli')
        thompson_sampling_bern(thompson_bandit, num_rounds=num_rounds, type_of_pull="monte_carlo", cap_prop=True)
