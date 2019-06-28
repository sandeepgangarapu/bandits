import numpy as np
from math import sqrt
from statsmodels.stats.power import TTestIndPower
from bandits.utils import treatment_outcome_grouping


def power_analysis(outcome_lis_of_lis, alpha=0.05, beta=0.1):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param alpha: significance level
    :param beta: 1=b = power
    :return: sample_size for EACH group
    """
    
    # find effect size of all arms
    effect_size_lis = []
    mean_control = np.mean(outcome_lis_of_lis[0])
    var_control = np.var(outcome_lis_of_lis[0])
    for arm in range(1, len(outcome_lis_of_lis)):
        # mean of each arm
        mean_arm = np.mean(outcome_lis_of_lis[arm])
        var_arm = np.var(outcome_lis_of_lis[arm])
        effect_size = mean_arm-mean_control/(sqrt((var_arm+var_control)/2))
        effect_size_lis.append(effect_size)
    # we calculate min effect so that the sample size is large enough to
    # detect the smallest effect of the arm in a/b testing
    min_effect_size = np.min(np.abs(effect_size_lis))
    # find sample size using power analysis if sample_size is none
    # this sample size is for 1 arm
    sample_size = TTestIndPower().solve_power(effect_size=min_effect_size,
                                              power=1-beta, alpha=alpha)
    sample_size = int(sample_size)
    return sample_size


def ab_testing(outcome_lis_of_lis, sample_size=None, post_allocation=False):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param sample_size: number of subjects to be allocated to EACH group
    :param post_allocation: whether remaining subjects should be allocated
    to winning arm
    :return: group assignment list, outcome list
    """
    print("---------------Running AB testing---------------")
    num_arms = len(outcome_lis_of_lis)
    if not sample_size:
        sample_size = min(power_analysis(outcome_lis_of_lis),
                          len(outcome_lis_of_lis[0]))
    
    # # we now subset so that we only run A/B testing until sample size runs out
    # ab_lis_of_lis = []
    # for group in outcome_lis_of_lis:
    #     ab_lis_of_lis.append(group[:sample_size])

    # simulating A/B testing
    group_assigned = []
    outcome = []
    # in this method we assign one by one to all treatment arms
    # alternatively and realize the outcomes until we are exhausted of subjects
    for subject in range(sample_size):
        for arm in range(num_arms):
            group_assigned.append(arm)
            outcome.append(outcome_lis_of_lis[arm][subject])
    
    ab_lis_of_lis = treatment_outcome_grouping(group_assigned, outcome)
    
    # allocation of remaining subjects post AB testing
    if post_allocation:
        # for now we will check for means and allocate the rest to the winning
        winning_arm = np.argmax([np.mean(lis) for lis in ab_lis_of_lis])
        group_assigned.extend([winning_arm for i in range(len(outcome_lis_of_lis[winning_arm][sample_size:]))])
        outcome.extend(outcome_lis_of_lis[winning_arm][sample_size:])
    return group_assigned, outcome

