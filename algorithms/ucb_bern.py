from utils import ucb_value_1
from bandit import Bandit


def ucb_bern(bandit, num_rounds, type_of_pull='single'):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""
    print("---------------Running UCB ---------------")
    # choose each action once:
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)

    ucb_rounds = num_rounds - (2 * bandit.num_arms)
    for round in range(ucb_rounds):
        # find UCB for all arms
        arm_max_ucb = ucb_value_1(bandit.num_arms, len(bandit.arm_pull_tracker),
                                      bandit.arm_pull_tracker,
                                      bandit.avg_reward_tracker)
        bandit.pull_arm(arm_max_ucb)
    print(bandit.arm_pull_tracker)
    return bandit

if __name__ == '__main__':
    # Define bandit
    num_rounds = 1000
    thompson_bandit = Bandit(name='thompson_sampling',
                             arm_means=[0.1, 0.2, 0.3],
                             dist_type='Bernoulli')
    ucb_bern(thompson_bandit, num_rounds=num_rounds, type_of_pull="monte_carlo")
