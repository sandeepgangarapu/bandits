from utils import thompson_arm_pull
import random
import numpy
import time

if __name__ == '__main__':
    start_time = time.time()
    a = thompson_arm_pull(mean_lis=[1,2], var_lis=[1,1], type_of_pull="monte_carlo")
    print(a)
    print("--- %s seconds ---" % (time.time() - start_time))
