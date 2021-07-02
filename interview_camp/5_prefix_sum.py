# Given an array of positive and negative integers,
# #find a subarray whose sum equals X.


# ESTCV
# [2,1,3,4,5] - 5 - [2,3] [6]
# [1,2] - 5 - None
# [] - None

def sum_to_x(a,X):
    prefix_sum = {}
    sum_so_far = 0
    for i in range(len(a)):
        sum_so_far += a[i]
        prefix_sum[sum_so_far] = i
        if sum_so_far == X:
            return a[0: i+1]
        elif sum_so_far-X in prefix_sum:
            return a[prefix_sum[sum_so_far-X]+1: i+1]
    return None

print(sum_to_x([2,4,-2,1,-3,5,-3], 5))
print(sum_to_x([], 5))
