from utils import ucb_value_naive


def ucb(bandit, num_rounds, type_of_pull='single'):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""
    print("---------------Running UCB ---------------")
    # choose each action once:
    for ite in range(2):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)
    
    ucb_rounds = num_rounds - (2*bandit.num_arms)
    for round in range(ucb_rounds):
        # find UCB for all arms
        arm_max_ucb = ucb_value_naive(bandit.num_arms, ucb_rounds,
                                      bandit.arm_pull_tracker,
                                      bandit.avg_reward_tracker)
        bandit.pull_arm(arm_max_ucb)
    return bandit

