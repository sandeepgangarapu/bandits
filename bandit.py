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
        self.arm_prop_tracker = [[] for i in range(num_arms)]
        self.propensity_tracker = []
        self.ipw_avg_reward_tracker = [0.0 for i in range(num_arms)]
        self.ipw_tracker = []
        self.aipw_tracker = []
        self.aipw_avg_reward_tracker = [0.0 for i in range(num_arms)]
        self.total_reward_tracker = [0.0 for i in range(num_arms)]
        self.max_reward = 0.0
        self.max_reward_arm = 0
        self.regret = []
        self.var_est_tracker = [0.0 for i in range(num_arms)]
        self.var_of_var_est_tracker = [0.0 for i in range(num_arms)]
        self.var_change_tracker = [0.0 for i in range(num_arms)]
        self.trt_effect_est_tracker = [0.0 for i in range(num_arms - 1)]
        self.trt_effect_var_est_tracker = [0.0 for i in range(num_arms - 1)]
        self.trt_effect_var_of_var_est_tracker = [0.0 for i in
                                                  range(num_arms - 1)]
    
    def pull_arm(self, arm_num, propensity=None):
        """pulling an just generated reward"""
        
        # Generate reward from that arm
        reward = np.random.choice(self.trt_dist_list[arm_num])
        # Do all updates after the arm is pulled

        if propensity is not None:
            self.update_after_pull(arm_num, reward, propensity)
        else:
            self.update_after_pull(arm_num, reward)

    def aipw_term_3_calculator(self, arm, rew_lis):
        """
        This is to calculate term 3 of aipw estimator
        We need to accountn for sum(mu)/T everytime the indicator function is zero.
        This function will do that
        :return: term_3
        """
        non_indicator_mu_tracker = []
        i=1
        for a in self.arm_tracker:
            if a==arm:
                i=i+1
            else:
                non_indicator_mu_tracker.append(np.mean(rew_lis[:i]))
        return non_indicator_mu_tracker


    def update_after_pull(self, arm_num, reward, propensity=None):
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
        if propensity is not None:
            self.propensity_tracker.append(propensity)
            self.arm_prop_tracker[arm_num].append(propensity)
            # IPW estimate. refer page 3 para 2 athey hadad
            ipw_est = np.array(self.arm_reward_tracker[arm_num]) / np.array(self.arm_prop_tracker[arm_num])
            ipw_est = ipw_est.sum()/len(self.arm_tracker)
            if ipw_est > 100000:
                print('hey')
            self.ipw_tracker.append(ipw_est)
            self.ipw_avg_reward_tracker[arm_num] = ipw_est.sum()/len(self.arm_tracker)

            # AIPW estimate. refer page 6 eq 2 athey hadad
            rew_tracker = self.arm_reward_tracker[arm_num].copy()
            # because the prior mean for all arms is initialized to 1 in
            # thompson sampling
            rew_tracker.insert(0,1)
            # AIPW term 2 mu should be mean est until the previous alloc H(t-1)
            rew_tracker = rew_tracker[:-1]
            mu = np.cumsum(rew_tracker)/np.arange(1, len(rew_tracker)+1)
            term_1 = (np.array(self.arm_reward_tracker[arm_num]) /
                        np.array(self.arm_prop_tracker[arm_num]))
            term_2 = (np.ones(self.arm_pull_tracker[arm_num])-(np.ones(self.arm_pull_tracker[arm_num])/np.array(self.arm_prop_tracker[arm_num])))*mu
            aipw_est = term_1 + term_2
            term_3 = self.aipw_term_3_calculator(arm_num, rew_tracker)
            aipw_est = (np.sum(aipw_est)+np.sum(term_3))/len(self.arm_tracker)
            self.aipw_tracker.append(aipw_est)
            self.aipw_avg_reward_tracker[arm_num] = aipw_est
        # Update max_reward_arms
        self.max_reward = np.amax(self.avg_reward_tracker)
        # note that this is a list
        self.max_reward_arm = np.argmax(self.avg_reward_tracker)
        self.regret.append(sum(self.arm_pull_tracker) * np.amax(
            self.avg_reward_tracker) - sum(
            self.total_reward_tracker))
        
        # var change tracker should come before var tracker
        self.var_change_tracker[arm_num] = abs(np.var(self.arm_reward_tracker[
                                                          arm_num], ddof =1) -
                                               self.var_est_tracker[
                                                   arm_num])
        
        self.var_est_tracker[arm_num] = np.var(
            self.arm_reward_tracker[arm_num], ddof =1)

        self.var_of_var_est_tracker[arm_num] = 2*self.var_est_tracker[
            arm_num]/(self.arm_pull_tracker[arm_num]-1)
        
        # if arm_num != 0:
        #     # treatment effect tracker has no control arm
        #     self.trt_effect_est_tracker[arm_num-1] = \
        #         self.avg_reward_tracker[arm_num] - self.avg_reward_tracker[0]
        #
        #     # treatment effect variance is sum of variance of outcome dist
        #     self.trt_effect_var_est_tracker[arm_num-1] = \
        #     self.var_est_tracker[arm_num] \
        #     + self.var_est_tracker[0]
        #
        #     # variance of variance est
        #     if self.arm_pull_tracker[0] and self.arm_pull_tracker[
        #             arm_num] !=0:
        #         self.trt_effect_var_of_var_est_tracker[arm_num-1] = \
        #         ((self.var_est_tracker[arm_num] ** 2) / (self.arm_pull_tracker[
        #         arm_num])) + ((self.var_est_tracker[0] ** 2) /
        #                           (self.arm_pull_tracker[0]))
        #     else:
        #         self.trt_effect_var_of_var_est_tracker[arm_num-1] = 0
        #
        # else:
        #     # we change trt effect estimates of every arm
        #     self.trt_effect_est_tracker = [
        #         self.avg_reward_tracker[arm] - self.avg_reward_tracker[0] for
        #         arm in range(1, self.num_arms)]
        #
        #     self.trt_effect_var_est_tracker = \
        #         [self.var_est_tracker[arm] + self.var_est_tracker[0]
        #          for arm in range(1, self.num_arms)]
        #
        #     if np.max(self.arm_pull_tracker) > 2:
        #         self.trt_effect_var_of_var_est_tracker = \
        #             [((self.var_est_tracker[arm] ** 2) /
        #               (self.arm_pull_tracker[arm])) +
        #              ((self.var_est_tracker[0] ** 2) /
        #               (self.arm_pull_tracker[0]))
        #              if self.arm_pull_tracker[0] and self.arm_pull_tracker[
        #                 arm] != 0 else 0
        #              for arm in range(1, self.num_arms)]
        #     else:
        #         self.trt_effect_var_of_var_est_tracker = [0 for arm in range(
        #             1, self.num_arms)]
