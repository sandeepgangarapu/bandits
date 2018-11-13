import numpy as np
from math import sqrt
import pandas as pd


def linucb(feature_list, target, treatment):
    alpha = 1  # Some parameter
    K = 3  # Num arms
    d = 26  # Num features
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
        if chosen_arm==treatment[round]:
            # Pull arm
            reward = target[round]
            A[chosen_arm] = np.add(A[chosen_arm],
                               np.outer(feature_vec, feature_vec))
            b[chosen_arm] = np.add(b[chosen_arm], np.array(feature_vec) * reward)
            output.append({'chosen_arm': chosen_arm, 'reward': reward,
                           'optimal': 1})
        else:
            reward = target[round]
            output.append({'chosen_arm': chosen_arm, 'reward': reward,
                           'optimal': 0})
    return output


if __name__ == '__main__':
    data = pd.read_csv('donor_input.csv')
    # shuffle the data
    data = data.sample(frac=1)
    features = data.iloc[:,:-3].values
    benefit = data['firm_gain'].values
    cost = data['firm_cost'].values
    group = data['test_group'].values
    utility = np.subtract(benefit, cost)
    result = linucb(features, target=utility, treatment=group)
    result = pd.DataFrame(result)
    result['cost'] = cost
    result.to_csv("donor_linucb.csv")