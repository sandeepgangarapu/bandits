# Given group allocation and outcomes, this func gives the outcomes of control and trtments allcoated ito those
# groups so fat
# Output is a list of list
# control = []
# trt = []
# if group[0] == 0:
#     control.append([outcome[0]])
# else:
#     trt.append([outcome[0]])
#
# for i in range(1, len(group)):
#     if group[i] == 0:
#         control.append(control[-1]+[outcome[i]])
#         trt.append(trt[-1])
#     else:
#         control.append(control[-1])
#         trt.append(trt[-1]+[outcome[i]])
#
# c_outcome = [outcome[i] for i in range(len(outcome)) if group[i]==0]
# t_outcome = [outcome[i] for i in range(len(outcome)) if group[i]==1]
#
#
#
# def vanilla_ab(x, y):
#     """
#     :param x: list of control outcomes
#     :param y: list of treatment outcomes
#     :return: group assignment list, outcome list
#     """
#     # we first calculate power
#     num = float(np.mean(y) - np.mean(x))
#     den = float(sqrt((np.var(x) + np.var(y)) / (2)))
#     effect_size = num / den
#     alpha = 0.05
#     pr = 0.80
#     sample_size = TTestIndPower().solve_power(effect_size=effect_size,
#                                               power=pr, alpha=alpha)
#     sample_size = int(sample_size)
#     print(sample_size)
#     # we now subset so that we only run A/B testing until sample size runs out
#     x_ab = x[:sample_size]
#     y_ab = y[:sample_size]
#
#     # after AB testing, we only run the winner
#     x_after = x[sample_size:]
#     y_after = y[sample_size:]
#
#     # simulating A/B testing
#     group_assigned = []
#     outcome = []
#     # in this method we assign one by one to treatment and control
#     # alternatively and realize the outcomes until we are exhausted of subjects
#     for i in range(len(x_ab)):
#         group_assigned.append(0)
#         outcome.append(x_ab[i])
#         group_assigned.append(1)
#         outcome.append(y_ab[i])
#
#     # for noe we will check for means and allocate the rest to the winning
#     # group irrespective of significance, but this should change
#
#     for j in range(len(x_after)):
#         if np.mean(x_ab) > np.mean(y_ab):
#             print("yay")
#             group_assigned.append(0)
#             outcome.append(x_after[j])
#         else:
#             group_assigned.append(1)
#             outcome.append(y_after[j])
#     return group_assigned, outcome
#
#
#
# def always_valid_p_value(theta0, tau, mean_1, mean_2, var, num_rounds):
#     X = []
#     Y = []
#     outcome = []
#     group = []
#     test_stat = []
#     p = [1]
#     for n in range(1, num_rounds + 1):
#         x_sample = np.random.normal(loc=mean_1, scale=var)
#         X.append(x_sample)
#         group.append(0)
#         outcome.append(x_sample)
#         y_sample = np.random.normal(loc=mean_2, scale=var)
#         Y.append(y_sample)
#         group.append(1)
#         outcome.append(y_sample)
#         ybar = np.mean(Y)
#         # print(ybar)
#         xbar = np.mean(X)
#         # print(xbar)
#         lmd = (sqrt(2 * var ** 2 / (2 * var ** 2 + n * tau ** 2)))
#         lmd = lmd * exp((n ** 2 * tau ** 2 * (ybar - (xbar - theta0)) ** 2) / (
#                     4 * var ** 2 * (
#                     2 * var ** 2 + n * tau ** 2)))
#
#         p_new = min([p[-1], 1 / lmd])
#         p.append(p_new)
#         test_stat.append(lmd)
#     print(test_stat)
#     print(p)
#
#
# def always_valid_p_value_emp(theta0, tau, x, y):
#     X = []
#     Y = []
#     outcome = []
#     group = []
#     test_stat = []
#     p = [1]
#     for n in range(len(x)):
#         x_sample = x[n]
#         X.append(x_sample)
#         group.append(0)
#         outcome.append(x_sample)
#         y_sample = y[n]
#         Y.append(y_sample)
#         group.append(1)
#         outcome.append(y_sample)
#         ybar = np.mean(Y)
#         xbar = np.mean(X)
#         var = np.var(x)
#         lmd = (sqrt(2 * var ** 2 / (2 * var ** 2 + n * tau ** 2)))
#         lmd = lmd * exp((n ** 2 * tau ** 2 * (ybar - (xbar - theta0)) ** 2) / (
#                 4 * var ** 2 * (
#                 2 * var ** 2 + n * tau ** 2)))
#
#         p_new = min([p[-1], 1 / lmd])
#         p.append(p_new)
#         test_stat.append(lmd)
#         if p_new < 0.05:
#             x = x[n:]
#             y = y[n:]
#             if np.mean(X) > np.mean(Y):
#                 for i in range(len(x[n:])):
#                     group.append(0)
#                     outcome.append(x[i])
#             else:
#                 for i in range(len(y[n:])):
#                     group.append(0)
#                     outcome.append(y[i])
#             return group, outcome, p
#         test_stat.append(lmd)
#     return group, outcome, p
#
#
# def always_valid_p(theta0, tau, X, Y):
#     p = [1]
#     n = len(X)
#     ybar = np.mean(Y)
#     xbar = np.mean(X)
#     var = np.var(Y)
#     lmd = sqrt(2 * (var ** 2) / (2 * (var ** 2) + n * (tau ** 2)))
#     sub_exp = round(((n ** 2) * (tau ** 2) * ((ybar - (xbar - theta0)) ** 2))
#                     / (4 * (var ** 2) * (2 * (var ** 2) + n * (tau ** 2))), 5)
#     try:
#         lmd = lmd * exp(sub_exp)
#         p_always = min([p[-1], 1 / lmd])
#     except OverflowError:
#         p_always = min([p[-1], 0.04])
#     return p_always
#
#
# # two tailed p value
# def normal_p_value(x1, x2, equal_var=False):
#     t, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
#     return p
#
#
# if __name__ == '__main__':
#     # Null hypothesis parameters
#     theta_0 = 0
#     tau = 1
#     always_valid_p_value(theta0=theta_0, tau=tau, mean_1=0, mean_2=0.1,
#                          var=1, num_rounds=1000)