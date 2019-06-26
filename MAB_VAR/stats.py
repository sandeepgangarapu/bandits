import numpy as np
from bandits.utils import treatment_outcome_grouping
from sklearn.metrics import mean_squared_error
from math import sqrt

def time_lis_of_lis(lis):
    """
    :param lis: list of anything [9,8,4,4,8,9,6]
    :return: list of lists = [[9],[9,8],[9,8,4],[9,8,4,4]......]
    """
    # Given outcome list, this func gives the outcomes so far in a list
    # Output is a list of list
    time_lis_of_lis = []
    for i in range(1, len(lis)):
        time_lis_of_lis.append(lis[:i])
    return time_lis_of_lis


def overall_stats(outcome):
    """
    :param outcome: list of outcomes as per allocation
    :return: dict of stats
    """
    outcome_time_lis_of_lis = time_lis_of_lis(outcome)
    avg = [np.mean(lis) for lis in outcome_time_lis_of_lis]
    var = [np.var(lis) for lis in outcome_time_lis_of_lis]
    return {'avg': avg, 'var': var}


def regret_outcome(group, outcome):
    """
    :param group: list of group numbers
    :param outcome: list of outcomes
    :return: regret and rmse of variance
    """
    
    regret = []
    for i in range(1, len(group)+1):
        grp = group[:i]
        out = outcome[:i]
        arm, outcome_lis_of_lis = treatment_outcome_grouping(grp, out,
                                                        group_outcome=True)
        avg = [np.mean(lis) for lis in outcome_lis_of_lis]
        reg = i*np.max(avg) - sum(out)
        regret.append(reg)
    
    return regret


def rmse_outcome(group, outcome, true_mean, true_var):
    """
    :param group: list of group numbers
    :param outcome: list of outcomes
    :param true_mean: list of true means if arms
    :param true_var: list of tru variances of arms
    :return: mean rmse and variance rmse lists
    """
    mean_rmse_lis = []
    var_rmse_lis = []
    for i in range(1, len(group)+1):
        arms, outcome_lis_of_lis = treatment_outcome_grouping(
            group[:i], outcome[:i], group_outcome=True)
        mean_lis = [np.mean(lis) for lis in outcome_lis_of_lis]
        var_lis = [np.var(lis) for lis in outcome_lis_of_lis]
        print(var_lis)
        mean_true = []
        var_true = []
        for arm in arms:
            mean_true.append(true_mean[arm])
            var_true.append(true_var[arm])
        mean_rmse = sqrt(mean_squared_error(mean_true, mean_lis))
        var_rmse = sqrt(mean_squared_error(var_true, var_lis))
        mean_rmse_lis.append(mean_rmse)
        var_rmse_lis.append(var_rmse)
    return mean_rmse_lis, var_rmse_lis