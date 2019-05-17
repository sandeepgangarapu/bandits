from math import sqrt, exp
import numpy as np


def always_valid_pvalue(x, y, theta0=0, tau=2):
    """
    :param x: list of control outcomes
    :param y: list of treatment outcomes
    :param theta0: difference of means in null hypothesis
    :param tau: variance of null hypothesis
    :return: p value
    """
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    total_var = var_y + var_x
    diff_mean = mean_y - mean_x
    n = len(x)
    term1 = sqrt((total_var) / (total_var + n * tau ** 2))
    term2 = ((n ** 2) * (tau ** 2) * ((diff_mean - theta0) ** 2)) / (
                2 * total_var * (
                total_var + (n * (tau ** 2))))
    lmd = term1 * exp(term2)
    p_val = 1 / lmd
    return p_val
