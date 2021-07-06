import numpy as np
from math import sqrt

class Bandit:
    
    def __init__(self, name, arm_means, arm_vars=None, dist_type='Normal'):
        '''
        :param name: name of the bandit
        :param arm_means: list of arm means
        :param arm_vars: list of arm variances
        '''
        self.name = name
        self.arm_means = arm_means
        self.arm_vars = arm_vars
        self.dist_type = dist_type
        # number of arms in bandit
        self.num_arms = len(arm_means)
        # tracker that tracks all the arm numbers that are pulled
        self.arm_tracker = []
        # tracking all the rewards observed
        self.reward_tracker = []
        # rewards are separated by arm
        self.arm_reward_tracker = [[] for i in range(self.num_arms)]
        # number of times each arm is pulled
        self.arm_pull_tracker = [0 for i in range(self.num_arms)]
        # mean reward of each arm
        self.avg_reward_tracker = [0.0 for i in range(self.num_arms)]
        # tracking the propensity with with an arm at that time is pulled
        self.propensity_tracker = []
        # This tracks propensity list of all arms at the time of irrespective of which arm is pulled
        self.prop_lis_tracker = []
        # propensity grouped by the arms
        self.arm_prop_tracker = [[] for i in range(self.num_arms)]
        # total reward in each arm
        self.total_reward_tracker = [0.0 for i in range(self.num_arms)]
        # empirical regret
        self.regret = []
        # estimate of sample variance
        self.var_est_tracker = [0.0 for i in range(self.num_arms)]
    
    def pull_arm(self, arm_num, prop_lis=None):
        """pulling an just generated reward"""
        global reward
        if self.dist_type == "Normal":
            # Generate reward from that arm
            reward = np.random.normal(self.arm_means[arm_num], sqrt(self.arm_vars[arm_num]))
        if self.dist_type == "HSN" or self.dist_type == "LSN" or self.dist_type == "ZSN":
            reward = self.arm_means[arm_num]+ np.random.uniform(-1, 1)
        
        # Do all updates after the arm is pulled
        if self.dist_type == "Bernoulli":
            reward = 1 if np.random.random() < self.arm_means[arm_num] else 0
        if self.dist_type == "HSN_bern" or self.dist_type == "LSN_bern" or \
                self.dist_type == "ZSN_bern":
            reward = 1 if np.random.random() < (self.arm_means[arm_num]+ \
                          np.random.uniform(-0.2, 0.2)) else 0
        if prop_lis is not None:
            self.update_after_pull(arm_num, reward, prop_lis)
        else:
            self.update_after_pull(arm_num, reward)

    def update_after_pull(self, arm_num, reward, prop_lis=None):
        """Do updates to bandit params after an arm is pulled"""
        # track arm pulled
        self.arm_tracker.append(arm_num)
        # Track reward
        self.reward_tracker.append(reward)
        # Track reward per arm
        self.arm_reward_tracker[arm_num].append(reward)
        # add the arm_no that is pulled
        self.arm_pull_tracker[arm_num] = self.arm_pull_tracker[arm_num] + 1
        # Add the reward to reward tracker
        self.total_reward_tracker[arm_num] = self.total_reward_tracker[
                                                 arm_num] + reward
        # update average reward
        self.avg_reward_tracker[arm_num] = self.total_reward_tracker[arm_num] / self.arm_pull_tracker[arm_num]
        self.regret.append(sum(self.arm_pull_tracker) * np.amax(
            self.avg_reward_tracker) - sum(
            self.total_reward_tracker))
        self.var_est_tracker[arm_num] = np.var(
            self.arm_reward_tracker[arm_num], ddof=1)
        if prop_lis is not None:
            self.propensity_tracker.append(prop_lis[arm_num])
            self.prop_lis_tracker.append(prop_lis)
