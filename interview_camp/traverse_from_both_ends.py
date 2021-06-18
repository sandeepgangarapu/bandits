# Given a sorted array in non-decreasing order, return an array of squares of each number, also in non-decreasing order. For example:

# [-4,-2,-1,0,3,5] -> [0,1,4,9,16,25]

def square_sort(A):
    i = 0
    j = len(A)-1
    k = len(A)-1
    new_A = [None]*len(A)
    if len(A) == 0:
        return None
    while i != j:
        if A[i]**2 < A[j]**2:
            new_A[k] = A[j]**2
            k = k-1
            j = j-1
        else:
            new_A[k] = A[i]**2
            k = k-1
            i = i+1
    new_A[k] = A[i]**2
    return new_A


print(square_sort([-4,-2,-1,0,3,5]))
print(square_sort([4]))
print(square_sort([]))




#Given an array of integers, find the continuous subarray, which when sorted, results in the entire array being sorted. For example:
# A = [0,2,3,1,8,6,9], result is the subarray [2,3,1,8,6]


def cont_subarray(A):
    ln = len(A)
    min_out = max(A)
    max_out = min(A)
    i = 0
    j = ln-1
    while i < ln-1:
        if A[i+1] < A[i] & A[i+1] < min_out:
            min_out = A[i+1]
        i = i+1

    while j >= 0:
        if A[j-1] > A[j] & A[j-1] > max_out:
            max_out = A[j-1]
        j = j-1


    for m in range(ln):
        if A[m] > min_out:
            min_index = m
            break
    for n in range(ln, 0, -1):
        if A[n] < max_out:
            max_index = n
            break

    return min_index, max_index


print(cont_subarray([[0,2,3,1,8,6,9]]))