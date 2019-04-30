from distributions import trt_dist_list
from scipy import stats

def always_valid_p_value():
    pass
    

# two tailed p value
def normal_p_value(x1, x2, equal_var=False):
    t, p = stats.ttest_ind(x1, x2, equal_var=equal_var)
    return p

