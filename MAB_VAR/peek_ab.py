import numpy as np
from bandits.peeking.always_valid_p import always_valid_pvalue
from bandits.utils import treatment_outcome_grouping

    
def peeking_ab_testing(outcome_lis_of_lis, theta0=0, tau=2, post_allocation=False):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param theta0: difference of means in null hypothesis
    :param tau: variance of null hypothesis
    :return: group assignment list, outcome list
    """
    print("---------------Running Peek AB testing---------------")
    num_arms = len(outcome_lis_of_lis)
    group_assigned = []
    outcome = []
    # generally we expect each arm to have equal no of users
    # but lets get the minimum length
    
    min_subjects = np.min([len(arm) for arm in outcome_lis_of_lis])
    # we initialize p vale to 1 and update it with newer p
    # there is pval for control but it will be 1
    p_tracker = [1 for i in range(num_arms)]
    for subject in range(min_subjects):
        # we allocate one to control if atleast one arm does not reach
        # significance
        # we do not check for control arm
        if np.max(p_tracker[1:])>0.05:
            group_assigned.append(0)
            outcome.append(outcome_lis_of_lis[0][subject])
        for arm in range(1, num_arms):
            if p_tracker[arm]>0.05:
                group_assigned.append(arm)
                outcome.append(outcome_lis_of_lis[arm][subject])
                x = outcome_lis_of_lis[0][:subject+1]
                y = outcome_lis_of_lis[arm][:subject+1]
                p_val = always_valid_pvalue(x=x, y=y, theta0=theta0, tau=tau)
                p_tracker[arm] = np.min([p_val, p_tracker[arm]])
        
        # the above code runs until all arms reach significance
    peek_outcome_lis_of_lis = treatment_outcome_grouping(group_assigned, outcome)

        
    # once that happens, we allocate the rest to the best group
    if post_allocation:
        if np.max(p_tracker[1:]) < 0.05:
            # for now we will check for means and allocate the rest to the winning
            winning_arm = np.argmax([np.mean(lis) for
                                             lis in peek_outcome_lis_of_lis])
            leftover_subjects = len(outcome_lis_of_lis[winning_arm]) - len(
                peek_outcome_lis_of_lis[winning_arm])
            group_assigned.extend(
                [winning_arm for i in range(leftover_subjects)])
            outcome.extend(outcome_lis_of_lis[winning_arm][::-1][
                           :leftover_subjects])
            return group_assigned, outcome

    return group_assigned, outcome