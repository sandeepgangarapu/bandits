#Given an array with n marbles colored Red, White or Blue, sort them so that marbles of the same color are adjacent, with the colors in the order Red, White and Blue.
#Assume the colors are given as numbers - 0 (Red), 1 (White) and 2 (Blue).
#For example, if A = [1,0,1,2,1,0,1,2], Output = [0,0,1,1,1,1,2,2].

# Examples
# [1,0,1,2,1,0,1,2], Output = [0,0,1,1,1,1,2,2]
# [1,2,1,2] - [1,1,2,2]
# [] - None
# [2,0] - [0,2]

# Solution:
# we will have boundaries on left and right to exclude min and max. we will traverse the array
# from left to right and keep swapping in order to get the order right

def sort_marbles(a):
    if len(a) == 0:
        return None
    left_p = 0
    right_p = len(a)-1
    trav_p=left_p
    while trav_p-1!=right_p:
        if a[trav_p] < 1:
            swap(a, trav_p, left_p)
            left_p = left_p + 1
            trav_p = trav_p + 1
        elif a[trav_p] > 1:
            swap(a, trav_p, right_p)
            right_p = right_p -1
        else:
            trav_p = trav_p + 1
    return a

def swap(a, i, j):
    k = a[i]
    a[i] = a[j]
    a[j] = k


print(sort_marbles([1,0,1,2,1,0,1,2]))
print(sort_marbles([1,2,1,2]))
print(sort_marbles([2,1,2]))
print(sort_marbles([]))
print(sort_marbles([1,3]))