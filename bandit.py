import numpy as np


class Bandit:
    
    def __init__(self, name, num_arms, trt_dist_list):
        self.name = name
        self.num_arms = num_arms
        self.trt_dist_list = trt_dist_list
        self.arm_tracker = []
        self.reward_tracker = []
        self.arm_reward_tracker = [[] for i in range(num_arms)]
        self.arm_pull_tracker = [0 for i in range(num_arms)]
        self.avg_reward_tracker = [0.0 for i in range(num_arms)]
        self.total_reward_tracker = [0.0 for i in range(num_arms)]
        self.max_reward = 0.0
        self.max_reward_arm = 0
        self.regret = []
        self.var_est_tracker = [0.0 for i in range(num_arms)]
        self.var_change_tracker = [0.0 for i in range(num_arms)]
        self.trt_effect_est_tracker = [0.0 for i in range(num_arms - 1)]
        self.trt_effect_var_est_tracker = [0.0 for i in range(num_arms - 1)]
        self.trt_effect_var_of_var_est_tracker = [0.0 for i in
                                                  range(num_arms - 1)]
    
    def pull_arm(self, arm_num):
        """pulling an just generated reward"""
        
        # Generate reward from that arm
        reward = np.random.choice(self.trt_dist_list[arm_num])
        # Do all updates after the arm is pulled
        self.update_after_pull(arm_num, reward)
    
    def update_after_pull(self, arm_num, reward):
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
        self.avg_reward_tracker[arm_num] = self.total_reward_tracker[arm_num] / \
                                           self.arm_pull_tracker[arm_num]
        # Update max_reward_arms
        self.max_reward = np.amax(self.avg_reward_tracker)
        # note that this is a list
        self.max_reward_arm = np.argmax(self.avg_reward_tracker)
        self.regret.append(sum(self.arm_pull_tracker) * np.amax(
            self.avg_reward_tracker) - sum(
            self.total_reward_tracker))
        
        # var change tracker should come before var tracker
        self.var_change_tracker[arm_num] = abs(np.var(self.arm_reward_tracker[
                                                          arm_num]) -
                                               self.var_change_tracker[
                                                   arm_num])
        
        self.var_est_tracker[arm_num] = np.var(
            self.arm_reward_tracker[arm_num])
        
        if arm_num != 0:
            # treatment effect tracker has no control arm
            self.trt_effect_est_tracker[arm_num-1] = \
                self.avg_reward_tracker[arm_num] - self.avg_reward_tracker[0]
            
            self.trt_effect_var_est_tracker[arm_num-1] = \
            self.var_est_tracker[arm_num] \
            + self.var_est_tracker[0]
            
            if self.arm_pull_tracker[0]-1 and self.arm_pull_tracker[
                    arm_num]-1 !=0:
                self.trt_effect_var_of_var_est_tracker[arm_num-1] = \
                ((self.var_est_tracker[arm_num] ** 2) / (self.arm_pull_tracker[
                arm_num]-1)) + ((self.var_est_tracker[0] ** 2) /
                                  (self.arm_pull_tracker[0]-1))
            else:
                self.trt_effect_var_of_var_est_tracker[arm_num-1] = 0
            
        else:
            # we change trt effect estimates of every arm
            self.trt_effect_est_tracker = [
                self.avg_reward_tracker[arm] - self.avg_reward_tracker[0] for
                arm in range(1, self.num_arms)]
            
            self.trt_effect_var_est_tracker = \
                [self.var_est_tracker[arm] + self.var_est_tracker[0]
                 for arm in range(1, self.num_arms)]
            
            if np.max(self.arm_pull_tracker) > 2:
                self.trt_effect_var_of_var_est_tracker = \
                    [((self.var_est_tracker[arm] ** 2) /
                      (self.arm_pull_tracker[arm]-1)) +
                     ((self.var_est_tracker[0] ** 2) /
                      (self.arm_pull_tracker[0]-1))
                     if self.arm_pull_tracker[0]-1 and self.arm_pull_tracker[
                        arm] - 1 != 0 else 0
                     for arm in range(1, self.num_arms)]
            else:
                self.trt_effect_var_of_var_est_tracker = [0 for arm in range(
                    1, self.num_arms)]
