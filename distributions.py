import numpy as np
import random

num_obs = 10000

random.seed(8153003)
control = np.random.normal(loc=1, scale=1.0, size=num_obs)
trt1 = np.random.normal(loc=1.5, scale=1.0, size=num_obs)
trt2 = np.random.normal(loc=2.0, scale=1.0, size=num_obs)
trt3 = np.random.normal(loc=2.5, scale=2.0, size=num_obs)

trt_dist_list = [control, trt1, trt2, trt3]
