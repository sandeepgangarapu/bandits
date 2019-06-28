from bandits.utils import treatment_outcome_grouping
from bandits.MAB_VAR.stats import rmse_outcome

group = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]
outcome = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]
arms, lis = treatment_outcome_grouping(group, outcome, group_outcome=True)
print(arms, lis)


mean_var, var_var = rmse_outcome(group, outcome, true_mean=[])