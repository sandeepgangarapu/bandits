import numpy as np
from math import sqrt
import pandas as pd


def linucb(feature_list, target, ite, alpha, cost):
    K = 3  # Num arms
    d = 14  # Num features
    num_rounds = len(feature_list)
    A = [np.identity(d) for i in range(K)]  # Identity matrix
    b = [np.zeros(d) for i in range(K)]
    output = []
    for round in range(num_rounds):
        # Observe a feature
        feature_vec = feature_list[round]
        p = []
        for i in range(K):
            theta = np.dot(np.linalg.inv(A[i]), b[i])
            pta = np.dot(theta, feature_vec) + alpha * sqrt(np.dot(np.dot(
                feature_vec, np.linalg.inv(A[i])), feature_vec))
            p.append(pta)
        chosen_arm = np.random.choice(np.argwhere(p == np.amax(p)).flatten().tolist())
        reward = target[round][chosen_arm]
        
        sub_cost = cost[round][chosen_arm]
        A[chosen_arm] = np.add(A[chosen_arm],
                           np.outer(feature_vec, feature_vec))
        b[chosen_arm] = np.add(b[chosen_arm], np.array(feature_vec) * reward)
        output.append({'chosen_arm': chosen_arm, 'reward': reward,
                       'ite': ite, 'alpha':alpha, 'cost':sub_cost})
    return output


if __name__ == '__main__':
    num_iterations = 15
    results = []
    alpha = 10
    utility = pd.read_csv('don_utility.csv')
    cost = pd.read_csv('don_cost.csv')
    for ite in range(1, num_iterations+1):
        # shuffle the data
        print(ite)
        sub_utility = utility[utility.ite==ite]
        sub_cost = cost[cost.ite==ite]
        features = sub_utility.iloc[:, 0:14].values
        sub_utility = sub_utility.iloc[:, 18:21].values
        sub_cost = sub_cost.iloc[:, 18:21].values
        result = linucb(features, target=sub_utility, ite=ite, alpha=alpha,
                        cost =sub_cost)
        result = pd.DataFrame(result)
        results.append(result)
        pd.concat(results).to_csv("donor_linucb_cost.csv", index=False)