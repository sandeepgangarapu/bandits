import numpy as np
from math import sqrt
import pandas as pd


def return_reward(arm, feature_vector):
    if arm==0:
        if feature_vector[0]==0 and feature_vector[1]==0:
            return np.random.normal(loc=4, scale=1.0, size=1)
        elif feature_vector[0]==0 and feature_vector[1]==1:
            return np.random.normal(loc=3, scale=1.0, size=1)
        elif feature_vector[0]==1 and feature_vector[1]==0:
            return np.random.normal(loc=2, scale=1.0, size=1)
        elif feature_vector[0]==1 and feature_vector[1]==1:
            return np.random.normal(loc=1, scale=1.0, size=1)
    else:
        if feature_vector[0]==0 and feature_vector[1]==0:
            return np.random.normal(loc=3, scale=1, size=1)
        elif feature_vector[0]==0 and feature_vector[1]==1:
            return np.random.normal(loc=2.8, scale=1, size=1)
        elif feature_vector[0]==1 and feature_vector[1]==0:
            return np.random.normal(loc=5, scale=1, size=1)
        elif feature_vector[0]==1 and feature_vector[1]==1:
            return np.random.normal(loc=6, scale=1, size=1)
    
    
if __name__ == '__main__':

    gender_vars = [1, 0]
    ethnicity_vars = [0, 1]
    
    alpha = 1 # Some parameter
    K = 2 # Num arms
    d = 2 # Num features
    num_rounds = 10000
    A = [np.identity(d) for i in range(K)] # Identity matrix
    b = [np.zeros(d) for i in range(K)]
    X = []
    arm_list = []
    reward_list = []
    for round in range(1, num_rounds+1):
        
        # Observe a feature
        gender = np.random.choice(gender_vars)
        ethnicity = np.random.choice(ethnicity_vars)
        feature_vec = [gender, ethnicity]
        p = []
        for i in range(K):
            theta = np.dot(np.linalg.inv(A[i]), b[i])
            pta = np.dot(theta, feature_vec)+alpha*sqrt(np.dot(np.dot(
                feature_vec, np.linalg.inv(A[i])), feature_vec))
            p.append(pta)
        chosen_arm = np.random.choice(np.argwhere(p==np.amax(p)).flatten().tolist())
        arm_list.append(chosen_arm)
        X.append(feature_vec)
        # Pull arm
        reward = return_reward(chosen_arm, feature_vec)[0]
        reward_list.append(reward)
        A[chosen_arm] = np.add(A[chosen_arm], np.outer(feature_vec, feature_vec))
        b[chosen_arm] = np.add(b[chosen_arm], np.array(feature_vec)*reward)
        
    data = pd.DataFrame(X, columns=['gender', 'ethnicity'])
    data['reward'] = reward_list
    data['chosen_arm'] = arm_list
    print(data.groupby(['gender', 'ethnicity', 'chosen_arm'], as_index=False)['reward'].mean())
    print(data.groupby(['gender', 'ethnicity', 'chosen_arm'], as_index=False).count())
