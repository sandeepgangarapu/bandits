# Given a sorted array A and a target T, find the target.
# If the target is not in the array, find the number closest to the target. '
# For example, if A = [2,3,5,8,9,11] and T = 7, return index of 8, i.e. return 3.

def return_closest_number(lis, target):
    start = 0
    end = len(lis)-1
    while start<=end:
        mid = int(start + ((end - start) / 2))
        if lis[mid] < target:
            start = mid+1
        elif lis[mid] > target:
            end = mid-1
        else:
            return mid
    return mid-1 if abs(target-lis[mid]) >= abs(target-lis[mid-1]) else mid

print(return_closest_number([1,3,5,8,8,9], target=6))
print(return_closest_number([2,3,5,8,9,11], target=7))