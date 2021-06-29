from bandit import Bandit
from utils import thompson_arm_pull_bern
from math import sqrt


def thompson_sampling_bern_batched(bandit, num_rounds, cap_prop=False,
                                   batch_size=100, type_of_pull='single'):
    """Function that reproduces the steps involved in Thompson sampling
    algorithm"""
    print("---------------Running Thompson Sampling Bern "
          "batched---------------")
    num_arms = bandit.num_arms
    prior_params = [[1, 1] for i in range(num_arms)]

    num_initial_pulls = int(batch_size/num_arms)
    for ite in range(num_initial_pulls):
        for arm in range(num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in
                                               range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)
            x = bandit.reward_tracker[-1]
            # We calculate the posterior parameters of the beta distribution
            prior_params[arm][0] += x
            prior_params[arm][1] += 1-x

    # beta bernoulli updation process

    for batch in range(int(num_rounds/batch_size)-1):
        arm_counts = [0 for i in range(num_arms)]
        rewards = [0 for i in range(num_arms)]
        # cap will be same for the entire batch
        cap = 0.1 / sqrt(((batch+1)*batch_size) + 1)
        for j in range(batch_size):
            # this is to cap probility of allocation as per Athey's algortihm
            if cap_prop:
                chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull, cap_prop=cap)
            else:
                chosen_arm, prop_lis = thompson_arm_pull_bern(param_lis=prior_params, type_of_pull=type_of_pull)
            bandit.pull_arm(chosen_arm, prop_lis)
            arm_counts[chosen_arm] += 1
            x = bandit.reward_tracker[-1]
            rewards[chosen_arm] += x
        
        # We calculate the posterior parameters of the beta distribution
        # after the batch is completely allocated
        for arm in range(num_arms):
            prior_params[arm][0] += rewards[arm]
            prior_params[arm][1] += arm_counts[arm] - rewards[arm]
            # print(prior_params[arm])
            
    return bandit


if __name__ == '__main__':
    # Define bandit
    for i in range(120):
        num_rounds = 2000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.5, 0.5, 0.5],
                                 dist_type='LSN_bern')
        thompson_sampling_bern_batched(thompson_bandit,
                               num_rounds=num_rounds, cap_prop=True,
                               type_of_pull='monte_carlo')
