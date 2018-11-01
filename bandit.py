import numpy as np


class Bandit:
    
    def __init__(self, name, num_arms, trt_dist_list):
        self.name = name
        self.num_arms = num_arms
        self.trt_dist_list = trt_dist_list
        self.arm_pull_tracker = [0 for i in range(num_arms)]
        self.total_reward_tracker = [0 for i in range(num_arms)]
        self.avg_reward_tracker = [0 for i in range(num_arms)]
        self.max_reward = 0
        self.max_reward_arm = 0
        self.regret = []
    
    def pull_arm(self, arm_num):
        """pulling an just generated reward"""
        
        # Generate reward from that arm
        reward = np.random.choice(self.trt_dist_list[arm_num])
        # Do all updates after the arm is pulled
        self.update_after_pull(arm_num, reward)
        
    def update_after_pull(self, arm_num, reward):
        """Do updates to bandit params after an arm is pulled"""

        # add the arm_no that is pulled
        self.arm_pull_tracker[arm_num] = self.arm_pull_tracker[arm_num]+1
        # Add the reward to reward tracker
        self.total_reward_tracker[arm_num] = self.total_reward_tracker[arm_num]+reward
        # update average reward
        self.avg_reward_tracker[arm_num] = self.total_reward_tracker[arm_num]/self.arm_pull_tracker[arm_num]
        # Update max_reward_arms
        self.max_reward = np.amax(self.avg_reward_tracker)
        # note that this is a list
        max_reward_arms = np.argwhere(self.avg_reward_tracker ==
                                      self.max_reward).flatten().tolist()
        # There might be two arms with max rewards. So we choose one at random
        self.max_reward_arm = np.random.choice(max_reward_arms)
        self.regret.append(sum(self.arm_pull_tracker)*np.amax(
            self.avg_reward_tracker) - sum(
            self.total_reward_tracker))
        

