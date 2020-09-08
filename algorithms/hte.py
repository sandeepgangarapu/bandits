from bandit import Bandit
import numpy as np
import random


def always_explore(bandit, num_rounds, num_arms):
    """Function that reproduces the steps involved in explore_first
    algorithm"""
    num_explore_each_arm = int(num_rounds / num_arms)
    for round in range(num_explore_each_arm):
        for arm in range(num_arms):
            bandit.pull_arm(arm)
    return bandit
