import numpy as np
from bandits.utils import treatment_outcome_grouping
from bandits.utils import ucb_value_naive
from math import sqrt


def treat_mixed_thompson(arm_means, arm_vars, num_subjects, perc_ab=0.2):
    """
    :param outcome_lis_of_lis: list of list of outcomes of all treatment groups
    :param perc_ab: percentage of times that the allocation must be made to
    A/B testing
    :return: group assignment list, outcome list
    """
    print("---------------Running Treat alloc Thompson testing---------------")
    num_arms = len(arm_means)
    group_assigned = []
    outcome = []
    # We initialize the variance
    var_tracker = [0 for i in range(num_arms)]
    trt_var_tracker = [var_tracker[i]+var_tracker[0] for i in range(
        1, num_arms)]
    
    var_change_tracker = [0 for i in range(num_arms)]
    trt_var_change_tracker = [0 for i in range(num_arms-1)]
    
    
    
    # allocate one subject to each arm (UCB rules)
    for arm in range(num_arms):
        group_assigned.append(arm)
        outcome.append(np.random.normal(loc=arm_means[arm],
                                        scale=sqrt(arm_vars[arm])))
        
        # everytime we pull arm we update var_tracker and var_change_tracker
        arms, out = treatment_outcome_grouping(group_assigned, outcome,
                                               all_arms=True,
                                               num_arms=num_arms)
        var = np.var(out[arm])
        var_change_tracker[arm] = abs(var_tracker[arm] - var)
        var_tracker[arm] = var
        
        if arm!=0:
            trt_var_change_tracker[arm-1] = abs(trt_var_tracker[arm-1]-(var_tracker[arm]+var_tracker[0]))
            trt_var_tracker[arm-1] = var_tracker[arm]+var_tracker[0]
        else:
            trt_var_change_tracker = [abs(trt_var_tracker[i]-(var_tracker[
                                                                  i+1]+var_tracker[0])) for i in range(num_arms-1)]
            trt_var_tracker = [var_tracker[i+1]+var_tracker[0] for i in range(num_arms-1)]
            
    # for now, lets assume all groups have same no. of subjects
    for subject in range(num_subjects - num_arms):
        # perc_ab of the time, we do AB testing otherwise bandits
        if np.random.uniform(0, 1) < perc_ab:
            # now we pick the arm that has highest variance change
            if np.max(trt_var_change_tracker) < 0.05:
                # do UCB again
                arm_pull_tracker = [group_assigned.count(arm) for arm in range(
                    num_arms)]
                # we find average reward of all arms so far
                # outcome lis of lis so far
                arms, out_lis_of_lis = treatment_outcome_grouping(
                    group_assigned,
                    outcome,
                    group_outcome=True)
                avg_reward_tracker = [np.mean(out_lis_of_lis[arm]) if arm in
                                                                      arms else 0
                                      for arm in range(num_arms)]
                ucb = ucb_value_naive(num_arms=num_arms,
                                      num_rounds=num_subjects,
                                      arm_pull_tracker=arm_pull_tracker,
                                      avg_reward_tracker=avg_reward_tracker)
                arm_max_ucb = np.argmax(ucb)
                group_assigned.append(arm_max_ucb)
                outcome.append(np.random.normal(loc=arm_means[arm_max_ucb],
                                                scale=sqrt(
                                                    arm_vars[arm_max_ucb])))
                # everytime we pull arm we update var_tracker and var_change_tracker
                arms, out = treatment_outcome_grouping(group_assigned, outcome,
                                                       all_arms=True,
                                                       num_arms=num_arms)
                var = np.var(out[arm_max_ucb])
                var_change_tracker[arm_max_ucb] = abs(
                    var_tracker[arm_max_ucb] - var)
                var_tracker[arm_max_ucb] = var
                if arm_max_ucb != 0:
                    trt_var_change_tracker[arm_max_ucb - 1] = abs(
                        trt_var_tracker[arm_max_ucb - 1] - (
                                    var_tracker[arm_max_ucb] + var_tracker[0]))
                    trt_var_tracker[arm_max_ucb - 1] = var_tracker[arm_max_ucb] + var_tracker[
                        0]
                else:
                    trt_var_change_tracker = [
                        abs(trt_var_tracker[i] - (var_tracker[
                                                      i + 1] + var_tracker[0]))
                        for i in range(num_arms - 1)]
                    trt_var_tracker = [var_tracker[i + 1] + var_tracker[0] for
                                       i in range(num_arms - 1)]
                continue
                
                # we find the arm that has highest treat effect change
                # we add one because trt_var_change_tracker has num_arms-1 values

            arm = np.argmax(trt_var_change_tracker) + 1
            
            # Then we decide whether we pull control or the treatment arm
            # we do that by proportionally picking according to the variance
            # change of those arms
            probability_of_control = var_change_tracker[
                                         0]/(var_change_tracker[0]+var_change_tracker[arm])
            arm = np.random.choice([arm,0], p=[1-probability_of_control,probability_of_control])
            
            group_assigned.append(arm)
            outcome.append(np.random.normal(loc=arm_means[arm],
                                            scale=sqrt(arm_vars[arm])))
            # everytime we pull arm we update var_tracker and var_change_tracker
            arms, out = treatment_outcome_grouping(group_assigned, outcome,
                                                   all_arms=True,
                                                   num_arms=num_arms)
            var = np.var(out[arm])
            var_change_tracker[arm] = abs(var_tracker[arm] - var)
            var_tracker[arm] = var
            if arm != 0:
                trt_var_change_tracker[arm - 1] = abs(
                    trt_var_tracker[arm - 1] - (
                                var_tracker[arm] + var_tracker[0]))
                trt_var_tracker[arm - 1] = var_tracker[arm] + var_tracker[0]
            else:
                trt_var_change_tracker = [
                    abs(trt_var_tracker[i] - (var_tracker[
                                                  i + 1] + var_tracker[0])) for
                    i in range(num_arms - 1)]
                trt_var_tracker = [var_tracker[i + 1] + var_tracker[0] for i in
                                   range(num_arms - 1)]


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
                                  num_rounds=num_subjects,
                                  arm_pull_tracker=arm_pull_tracker,
                                  avg_reward_tracker=avg_reward_tracker)
            arm_max_ucb = np.argmax(ucb)
            group_assigned.append(arm_max_ucb)
            outcome.append(np.random.normal(loc=arm_means[arm_max_ucb],
                                            scale=sqrt(arm_vars[arm_max_ucb])))
            # everytime we pull arm we update var_tracker and var_change_tracker
            arms, out = treatment_outcome_grouping(group_assigned, outcome,
                                                   all_arms=True,
                                                   num_arms=num_arms)
            var = np.var(out[arm_max_ucb])
            var_change_tracker[arm_max_ucb] = abs(
                var_tracker[arm_max_ucb] - var)
            var_tracker[arm_max_ucb] = var
            if arm_max_ucb != 0:
                trt_var_change_tracker[arm_max_ucb - 1] = abs(
                    trt_var_tracker[arm_max_ucb - 1] - (
                            var_tracker[arm_max_ucb] + var_tracker[0]))
                trt_var_tracker[arm_max_ucb - 1] = var_tracker[arm_max_ucb] + \
                                                   var_tracker[
                                                       0]
            else:
                trt_var_change_tracker = [
                    abs(trt_var_tracker[i] - (var_tracker[
                                                  i + 1] + var_tracker[0]))
                    for i in range(num_arms - 1)]
                trt_var_tracker = [var_tracker[i + 1] + var_tracker[0] for
                                   i in range(num_arms - 1)]
                
    return group_assigned, outcome


