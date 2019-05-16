from bandits.distributions import trt_dist_list
from scipy import stats
from math import sqrt, exp
import numpy as np


def always_valid_p_value(theta0, tau, mean_1, mean_2, var, num_rounds):
    X = []
    Y = []
    outcome = []
    group = []
    test_stat = []
    p = [1]
    for n in range(1, num_rounds+1):
        x_sample = np.random.normal(loc=mean_1, scale=var)
        X.append(x_sample)
        group.append(0)
        outcome.append(x_sample)
        y_sample = np.random.normal(loc=mean_2, scale=var)
        Y.append(y_sample)
        group.append(1)
        outcome.append(y_sample)
        ybar = np.mean(Y)
        #print(ybar)
        xbar = np.mean(X)
        #print(xbar)
        lmd = (sqrt(2*var**2/(2*var**2+n*tau**2)))
        lmd = lmd*exp((n**2*tau**2*(ybar-(xbar-theta0))**2)/(4*var**2*(
                2*var**2+n*tau**2)))
        
        p_new = min([p[-1], 1/lmd])
        p.append(p_new)
        test_stat.append(lmd)
    print(test_stat)
    print(p)





def always_valid_p_value_emp(theta0, tau, x, y):
    X = []
    Y = []
    outcome = []
    group = []
    test_stat = []
    p = [1]
    for n in range(len(x)):
        x_sample = x[n]
        X.append(x_sample)
        group.append(0)
        outcome.append(x_sample)
        y_sample = y[n]
        Y.append(y_sample)
        group.append(1)
        outcome.append(y_sample)
        ybar = np.mean(Y)
        xbar = np.mean(X)
        var = np.var(x)
        lmd = (sqrt(2 * var ** 2 / (2 * var ** 2 + n * tau ** 2)))
        lmd = lmd * exp((n ** 2 * tau ** 2 * (ybar - (xbar - theta0)) ** 2) / (
                    4 * var ** 2 * (
                    2 * var ** 2 + n * tau ** 2)))
        
        p_new = min([p[-1], 1 / lmd])
        p.append(p_new)
        test_stat.append(lmd)
        if p_new < 0.05:
            x = x[n:]
            y = y[n:]
            if np.mean(X) > np.mean(Y):
                for i in range(len(x[n:])):
                    group.append(0)
                    outcome.append(x[i])
            else:
                for i in range(len(y[n:])):
                    group.append(0)
                    outcome.append(y[i])
            return group, outcome, p
        test_stat.append(lmd)
    return group, outcome, p




def always_valid_p(theta0, tau, X, Y):
    p = [1]
    n = len(X)
    ybar = np.mean(Y)
    xbar = np.mean(X)
    var = np.var(Y)
    lmd = sqrt(2 * (var ** 2) / (2 * (var ** 2) + n * (tau ** 2)))
    sub_exp = round(((n ** 2) * (tau ** 2) * ((ybar - (xbar - theta0)) ** 2))
                    / (4 * (var ** 2) * (2 * (var ** 2) + n * (tau ** 2))),5)
    try:
        lmd = lmd * exp(sub_exp)
        p_always = min([p[-1], 1 / lmd])
    except OverflowError:
        p_always = min([p[-1], 0.04])
    return p_always


# two tailed p value
def normal_p_value(x1, x2, equal_var=False):
    t, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
    return p


if __name__ == '__main__':
    # Null hypothesis parameters
    theta_0 = 0
    tau = 1
    always_valid_p_value(theta0=theta_0, tau=tau, mean_1=0, mean_2=0.1,
                         var=1, num_rounds=1000)