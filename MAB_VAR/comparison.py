from bandits.MAB_VAR.ab_testing import ab_testing
from bandits.MAB_VAR.peek_ab import peeking_ab_testing
from bandits.MAB_VAR.vanilla_mix import vanilla_mixed_UCB
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.bandit import Bandit
from bandits.utils import create_distributions_vanilla
from bandits.MAB_VAR.stats import overall_stats


if __name__ == '__main__':
    
    num_arms = 10
    outcome_lis_of_lis = create_distributions_vanilla(num_arms)
    ab_group, ab_outcome = ab_testing(outcome_lis_of_lis, post_allocation=True)
    peek_group, peek_outcome = peeking_ab_testing(outcome_lis_of_lis,
                                                  post_allocation=True)
    mix_group, mix_outcome = vanilla_mixed_UCB(outcome_lis_of_lis)
    ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                        trt_dist_list=outcome_lis_of_lis)
    ucb_bandit = ucb_naive(bandit=ucb_bandit,
                           num_rounds=len(outcome_lis_of_lis[0]),
                           num_arms=num_arms)
    ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
    ab, peek, ucb, mix = map(overall_stats, [ab_outcome, peek_outcome,
                                             ucb_outcome, mix_outcome])
    print(ab)

