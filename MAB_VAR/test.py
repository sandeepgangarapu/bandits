from bandits.utils import treatment_outcome_grouping
from bandits.MAB_VAR.stats import rmse_outcome

group = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]
outcome = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]



num_arms = 10
arm_pull_tracker = [group.count(arm) for arm in range(
                num_arms)]

print(arm_pull_tracker)
arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group, outcome, group_outcome=False, all_arms=True, num_arms = 10)

print(arms, outcome_lis_of_lis)
