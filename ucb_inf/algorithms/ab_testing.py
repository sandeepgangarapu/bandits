import numpy as np
from math import sqrt
from statsmodels.stats.power import TTestIndPower
from bandits.utils import treatment_outcome_grouping


def power_analysis(arm_means, arm_vars, alpha=0.05, beta=0.1):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param alpha: significance level
    :param beta: 1=b = power
    :return: sample_size for EACH group
    """
    
    # find effect size of all arms
    num_arms = len(arm_means)
    effect_size_lis = []
    for arm in range(1, num_arms):
        effect_size = arm_means[arm]-arm_means[0]/(sqrt((arm_vars[arm]
            +arm_vars[0])/2))
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


def ab_testing(arm_means, arm_vars, num_subjects, sample_size=None,
               post_allocation=False):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param sample_size: number of subjects to be allocated to EACH group
    :param post_allocation: whether remaining subjects should be allocated
    to winning arm
    :return: group assignment list, outcome list
    """
    num_arms = len(arm_means)
    num_subjects_each = int(num_subjects/num_arms)
    print("---------------Running AB testing---------------")
    if not sample_size:
        sample_size = min(power_analysis(arm_means, arm_vars),
                          num_subjects_each)
    
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
            outcome.append(np.random.normal(loc=arm_means[arm],
                                            scale=sqrt(arm_vars[arm])))
    
    ab_lis_of_lis = treatment_outcome_grouping(group_assigned, outcome)
    
    # allocation of remaining subjects post AB testing
    if post_allocation:
        subjects_left = num_subjects-(sample_size*num_arms)
        # for now we will check for means and allocate the rest to the winning
        winning_arm = np.argmax([np.mean(lis) for lis in ab_lis_of_lis])
        group_assigned.extend([winning_arm for i in range(subjects_left)])
        outcome.extend([np.random.normal(loc=arm_means[winning_arm],
                                         scale=sqrt(arm_vars[winning_arm]))
                        for i in range(subjects_left)])
    return group_assigned, outcome

