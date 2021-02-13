import numpy as np
from bandit import Bandit
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def power_analysis(arm_means, alpha=0.05, beta=0.1):
    """
    :param alpha: significance level
    :param beta: 1-b = power
    :return: sample_size for EACH group
    """

    # find effect size of all arms
    num_arms = len(arm_means)
    effect_size_lis = []
    for arm1 in range(num_arms):
        for arm2 in range(num_arms):
            if arm1 != arm2:
                effect_size = proportion_effectsize(arm_means[arm1], arm_means[arm2])
                effect_size_lis.append(effect_size)
    # TODO: we calculate effect sizes twice for every pair of arms. This is a very very small
    # gain in efficiency but a gain nonetheless

    # we calculate min effect so that the sample size is large enough to
    # detect the smallest effect of the arm in a/b testing
    min_effect_size = np.min(np.abs(effect_size_lis))
    # find sample size using power analysis if sample_size is none
    # this sample size is for 1 arm
    try:
        sample_size = NormalIndPower().solve_power(effect_size=min_effect_size,
                                                  power=1 - beta, alpha=alpha, ratio=1)
    except ValueError:
        sample_size = 1000000
    sample_size = int(sample_size)
    return sample_size


def ab_testing_bern(bandit, num_rounds, sample_size=None,
               post_allocation=True):
    """
    :param bandit: ab_bandit
    :param num_rounds: total rounds for simulation
    :param arm_means: arm_means
    :param arm_vars: arm_vars
    :param sample_size: allocation to each arm
    :param post_allocation: if there should be max allocation after pure ab
    :return: bandit
    """
    arm_means = bandit.arm_means

    num_rounds_each = int(num_rounds / bandit.num_arms)
    print("---------------Running AB testing Bern---------------")
    if not sample_size:
        # max we can do is num_rounds_each
        sample_size = min(power_analysis(arm_means),
                          num_rounds_each)

    # simulating A/B testing
    if sample_size > num_rounds_each:
        sample_size = num_rounds_each
    for round in range(sample_size):
        for arm in range(bandit.num_arms):
            bandit.pull_arm(arm)

    # allocation of remaining subjects post AB testing
    if post_allocation:
        remaining_rounds = num_rounds - (sample_size * bandit.num_arms)
        # for now we will check for means and allocate the rest to the winning
        winning_arm = np.argmax(bandit.avg_reward_tracker)
        for round in range(remaining_rounds):
            bandit.pull_arm(winning_arm)
    print(bandit.arm_pull_tracker)
    return bandit

if __name__ == '__main__':
    # Define bandit
    for i in range(10):
        num_rounds = 1000
        thompson_bandit = Bandit(name='thompson_sampling',
                                 arm_means=[0.1, 0.3, 0.5],
                                 dist_type='Bernoulli')
        ab_testing_bern(thompson_bandit, num_rounds=num_rounds)
