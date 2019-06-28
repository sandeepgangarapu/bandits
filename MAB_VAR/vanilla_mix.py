import numpy as np
from bandits.utils import treatment_outcome_grouping
from bandits.utils import ucb_value_naive


def vanilla_mixed_UCB(outcome_lis_of_lis, perc_ab=0.2):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param perc_ab: percentage of times that the allocation must be made to
    A/B testing
    :return: group assignment list, outcome list
    """
    print("---------------Running mix UCB testing---------------")

    num_arms = len(outcome_lis_of_lis)
    group_assigned = []
    outcome = []
    
    # we do this because we will remove elements as and when we assign so we
    # acceess the first element always
    
    temp_outcome_lis_of_lis = outcome_lis_of_lis.copy()

    # allocate one subject to each arm (UCB rules)
    for arm in range(num_arms):
        group_assigned.append(arm)
        outcome.append(temp_outcome_lis_of_lis[arm][0])
        temp_outcome_lis_of_lis[arm] = temp_outcome_lis_of_lis[arm][1:]
        
    # for now, lets assume all groups have same no. of subjects
    for subject in range(len(outcome_lis_of_lis[0])-num_arms):
        # perc_ab of the time, we do AB testing otherwise bandits
        if np.random.uniform(0, 1) < perc_ab:
            # now that we are inside A/B testing, we randomly pick the arm
            # and assign a subject to that arm
            arm = np.random.randint(0, num_arms)
            outcome.append(temp_outcome_lis_of_lis[arm][0])
            temp_outcome_lis_of_lis[arm] = temp_outcome_lis_of_lis[arm][1:]
        else:
            arm_pull_tracker = [group_assigned.count(arm) for arm in range(
                num_arms)]
            # we find average reward of all arms so far
            # outcome lis of lis so far
            arms, out_lis_of_lis = treatment_outcome_grouping(group_assigned,
                                                        outcome,
                                                        group_outcome=True)
            avg_reward_tracker = [np.mean(out_lis_of_lis[arm]) if arm in
                                                                  arms else 0
                                  for arm in range(num_arms)]
            ucb = ucb_value_naive(num_arms=num_arms,
                                  num_rounds=len(outcome_lis_of_lis[0]),
                                  arm_pull_tracker=arm_pull_tracker,
                                  avg_reward_tracker=avg_reward_tracker)
            arm_max_ucb = np.argmax(ucb)
            group_assigned.append(arm_max_ucb)
            outcome.append(temp_outcome_lis_of_lis[arm_max_ucb][0])
            temp_outcome_lis_of_lis[arm_max_ucb] = temp_outcome_lis_of_lis[
                arm_max_ucb][1:]
            
    return group_assigned, outcome


