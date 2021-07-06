def find_if_number_present(lis, num):
    start = 0
    end = len(lis)-1
    while start <= end:
        mid = int(start + ((end - start) / 2))
        if lis[mid] < num:
            start = mid+1
        elif lis[mid] > num:
            end = mid-1
        else:
            return mid
    return None

# [1,2,3,4] - 2 should give 1
# [1,2,3] - 2 should give 1
# [1,3,4] - 2 should give None

#print(find_if_number_present([1,2,3,4], 2))
#print(find_if_number_present([1,2,3,], 2))
print(find_if_number_present([1,3,4], 2))

def dup_find_if_number_present(lis, num):
    start = 0
    end = len(lis)-1
    while start <= end:
        mid = int(start + ((end - start) / 2))
        if lis[mid] < num:
            start = mid+1
        elif lis[mid] > num:
            end = mid-1
        else:
            while lis[mid-1]==num:
                mid=mid-1
            return mid
    return None

# [1,2,2,2,3,4] - 2 should give 1
# [1,2,2,3,4,5,6,7] - 2 should give 1
# [1,3,4] - 2 should give None
print(dup_find_if_number_present([1,2,3,4], 2))
print(dup_find_if_number_present([1,2,3,], 2))
print(dup_find_if_number_present([1,3,4], 2))


def dup_find_if_number_present_log(lis, num):
    start = 0
    end = len(lis)-1
    while start <= end:
        mid = int(start + ((end - start) / 2))
        if lis[mid] < num:
            start = mid+1
        elif lis[mid] > num:
            end = mid-1
        else:
            while lis[mid-1]==num:
                mid_2 = int(start + ((mid - start) / 2))
                if lis[mid_2] < num:
                    start = mid_2 + 1
                elif lis[mid_2] > num:
                    mid = mid_2 - 1
            return mid
    return None