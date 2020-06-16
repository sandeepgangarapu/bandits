from bandits.bandit import Bandit
from bandits.utils import trt_dist_list, num_obs
from bandits.utils import ucb_value_naive


def ucb(bandit, num_rounds, type_of_pull='single'):
    """Function that reproduces the steps involved in ucb_naive
    algorithm"""
    #print("---------------Running UCB ---------------")
    # choose each action once:
    for ite in range(2):
        for arm in range(bandit.num_arms):
            if type_of_pull == 'monte_carlo':
                bandit.pull_arm(arm, prop_lis=[1 if i == arm else 0 for i in range(bandit.num_arms)])
            else:
                bandit.pull_arm(arm)
    
    ucb_rounds = num_rounds - (2*bandit.num_arms)
    for round in range(ucb_rounds):
        # find UCB for all arms
        if type_of_pull == 'monte_carlo':
            arm_max_ucb, prop_lis = ucb_value_naive(bandit.num_arms, ucb_rounds,
                                                    bandit.arm_pull_tracker,
                                                    bandit.avg_reward_tracker,
                                                    bandit.var_est_tracker,
                                                    type_of_pull='monte_carlo')
        else:
            arm_max_ucb = ucb_value_naive(bandit.num_arms, ucb_rounds,
                                          bandit.arm_pull_tracker,
                                          bandit.avg_reward_tracker,
                                          bandit.var_est_tracker)

        # Pull the arm with max ucb
        if type_of_pull == 'monte_carlo':
            bandit.pull_arm(arm_max_ucb, prop_lis)
        else:
            bandit.pull_arm(arm_max_ucb)
    return bandit


if __name__ == '__main__':
    # Define bandit
    num_arms = 4
    num_rounds = num_obs
    trt_dist_lis = trt_dist_list[:num_arms]
    ucb_bandit = Bandit(name='ucb_naive',
                        num_arms=num_arms,
                        trt_dist_list=trt_dist_lis)
    ucb(bandit=ucb_bandit,
        num_rounds=num_rounds)
