import numpy as np
import pandas as pd
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.algorithms.ucb_peek import ucb_peek
from bandits.bandit import Bandit
from bandits.peeking.always_valid_p import always_valid_p_value_emp
from bandits.utils import ucb_value_naive
import pickle
from bandits.peeking.utils import create_distributions_custom


def stats(lis_of_lis, agg):
    # Given a list of lists, this func return the value of aggregate function applied.
    # This is basically mutate function in R df munging.
    avg = []
    var = []
    cnt = []
    #
    for i in range(1, len(lis_of_lis)):
        try:
            if agg == 'avg':
                avg.append(np.mean(lis_of_lis[i]))
            if agg == 'var':
                var.append(np.var(lis_of_lis[i]))
            if agg == 'cnt':
                cnt.append(len(lis_of_lis[i]))
        except:
            pass
    if agg == 'avg':
        return avg
    if agg == 'cnt':
        return cnt
    if agg == 'var':
        return var


def split_outcome_lists(group, outcome):
    # Given group allocation and outcomes, this func gives the outcomes of control and trtments allcoated ito those
    # groups so fat
    # Output is a list of list
    control = [[0]]
    trt = [[0]]
    if group[0] == 0:
        control[0] = [outcome[0]]
    else:
        trt[0] = [outcome[0]]
    for i in range(1, len(group)):
        if group[i] == 0:
            control.append(control[-1] + [outcome[i]])
            trt.append(trt[-1])
        else:
            control.append(control[-1])
            trt.append(trt[-1] + [outcome[i]])
    return control, trt


def outcome_list(outcome):
    # Given group allocation and outcomes, this func gives the outcomes so far in a list
    # Output is a list of list
    lis_of_lis = [[outcome[0]]]
    for i in range(1, len(outcome)):
        lis_of_lis.append(lis_of_lis[-1] + [outcome[i]])
    return lis_of_lis


if __name__ == '__main__':
    N = 2000  # Number of subjects that can be allocated in each group
    # analysis output, so we can read it directly to plot
    out_file = "ucb_peek.csv"
    # Set seed for replicability
    np.random.seed(seed=123)
    # Create distributions
    num_arms = 10
    arm_means = np.arange(0.0, 1.0, 0.1)
    arm_vars = [1 for i in range(len(arm_means))]
    dist_list = create_distributions_custom(arm_means, arm_vars)
    # Switches to turn on and off for various analysis
    run_simulations = True
    run_plots = True
    all_grps = True
    seperate = False
    agg_type = "avg"
    if run_simulations:
        # Now we call all the algorithms that can simulate various allocation
        # procedures
        # calling ucb naive bandit algorithm
        ucb_vanilla_bandit = Bandit(name='ucb_van',
                            num_arms=num_arms,
                            trt_dist_list=dist_list)
        ucb_peek_bandit = Bandit(name='ucb_peek',
                            num_arms=num_arms,
                            trt_dist_list=dist_list)
        ucb_vanilla = ucb_naive(bandit=ucb_vanilla_bandit, num_rounds=N,
                                num_arms=num_arms)
        ucb_p = ucb_peek(bandit=ucb_peek_bandit, num_rounds=N, num_arms=num_arms)
        ucb_v_group, ucb_v_outcome = ucb_vanilla.arm_tracker, \
                                  ucb_vanilla.reward_tracker
        ucb_p_group, ucb_p_outcome = ucb_p.arm_tracker, \
                                     ucb_p.reward_tracker
        df = {"ucb_v_group":ucb_v_group, "ucb_v_outcome":ucb_v_outcome,
              "ucb_p_group":ucb_p_group, "ucb_p_outcome":ucb_p_outcome}
        df = pd.DataFrame.from_dict(df, orient='index')
        df = df.transpose()
        df['time'] = df.index
        df.to_csv(out_file, index=False)
