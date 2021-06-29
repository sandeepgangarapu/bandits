# ESTCV

# Given a String, find the longest substring with unique characters.

# Q: Will it have spaces?
# Q: Return substring, len or indices?

# solution

# test cases
# 'mynameissandeep' - 'ynameis'
# 'sandy' - 'sandy'


def longest_substring(stg):
    long_min_ind = 0
    long_max_ind = 1
    min_ind = 0
    max_ind = 1
    unique_str_set = {stg[0]}
    for i in range(1, len(stg)):
        if stg[i] in unique_str_set:
            while stg[min_ind] != stg[i]:
                min_ind += 1
            min_ind += 1
        else:
            unique_str_set.add(stg[i])
        max_ind += 1
        if max_ind - min_ind > long_max_ind - long_min_ind:
            long_max_ind = max_ind
            long_min_ind = min_ind
    return stg[long_min_ind:long_max_ind]


print(longest_substring('whatwhywhere'))
print(longest_substring('sandy'))

