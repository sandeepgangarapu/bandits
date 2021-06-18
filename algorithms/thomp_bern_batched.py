from bandit import Bandit
from utils import thompson_arm_pull_bern
from math import sqrt


def thompson_sampling_bern_batched(bandit, num_rounds, cap_prop=False,
                                   batch_size=100, type_of_pull='single'):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    print("---------------Running Thompson Sampling Bern "
          "batched---------------")

    # we use 2 in order to calculate sample variance
    num_arms = bandit.num_arms
    num_initial_pulls = int(batch_size/num_arms)
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)

    # beta bernoulli updation process
    prior_params = [[1, 1] for i in range(num_arms)]

    for batch in range(int(num_rounds/batch_size)-1):
        arm_counts = [0 for i in range(num_arms)]
        rewards = [0 for i in range(num_arms)]
        for j in range(batch_size):
            if type_of_pull == 'monte_carlo':
                chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
                bandit.pull_arm(chosen_arm, prop_lis=prop_lis)
            else:
                chosen_arm = thompson_arm_pull_bern(param_lis=prior_params)
                bandit.pull_arm(chosen_arm)
            arm_counts[chosen_arm] += 1
            x = bandit.reward_tracker[-1]
            rewards[chosen_arm] += x
        
        # We calculate the posterior parameters of the beta distribution
        # after the batch is completely allocated
        for arm in range(num_arms):
            prior_params[arm][0] += rewards[arm]
            prior_params[arm][1] += arm_counts[arm] - rewards[arm]

    return bandit


if __name__ == '__main__':
    # Define bandit
    for i in range(10):
        num_rounds = 1000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.1, 0.2, 0.3],
                                 dist_type='Bernoulli')
        thompson_sampling_bern_batched(thompson_bandit,
                                       num_rounds=num_rounds, cap_prop=False)
