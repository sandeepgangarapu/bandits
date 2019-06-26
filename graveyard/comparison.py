import numpy as np
import pandas as pd
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.bandit import Bandit
from bandits.peeking.always_valid_p import always_valid_p_value_emp
from bandits.utils import ucb_value_naive
from math import sqrt
import pickle


def stats(lis_of_lis, agg, target=None):
    # Given a list of lists, this func return the value of aggregate function applied.
    # This is basically mutate function in R df munging.
    avg = []
    var =[]
    cnt = []
    sum = []
    rse= []
    #
    for i in range(1,len(lis_of_lis)):
        try:
            if agg == 'avg':
                avg.append(np.mean(lis_of_lis[i]))
            if agg == 'var':
                var.append(np.var(lis_of_lis[i]))
            if agg == 'cnt':
                cnt.append(len(lis_of_lis[i]))
            if agg == 'sum':
                sum.append(np.sum(lis_of_lis[i]))
            if agg == 'rse':
                rse.append(sqrt((np.var(lis_of_lis[i])-target)**2))
        except:
            pass
    if agg=='avg':
        return avg
    if agg=='cnt':
        return cnt
    if agg=='var':
        return var
    if agg=='sum':
        return sum
    if agg=='rse':
        return rse




def outcome_list(outcome):
    # Given group allocation and outcomes, this func gives the outcomes so far in a list
    # Output is a list of list
    lis_of_lis = []
    for i in range(1, len(outcome)):
        lis_of_lis.append(outcome[:i])
    return lis_of_lis




def mixed_alg(x, y, perc_ab):
    outcome = [x[0], y[0]]
    X = [x[0]]
    Y = [y[0]]
    x = x[1:]
    y = y[1:]
    group = [0,1]
    
    for n in range(1, len(x) + 1):
        # 30% of the time, we do AB testing otherwise bandits
        if np.random.uniform(0, 1) < perc_ab:
            # now that we are inside A/B testing, 50% of times we allocate to X and Y
            if np.random.uniform(0, 1) < 0.5:
                group.append(0)
                X.append(x[0])
                outcome.append(x[0])
                x = x[1:]
            else:
                group.append(1)
                Y.append(y[0])
                outcome.append(y[0])
                y = y[1:]
        else:
            ucb = ucb_value_naive(2, 1000, [(np.array(group) == 0).sum(),
                                        (np.array(group) == 1).sum()],
                    [np.mean(X), np.mean(Y)])
            if ucb[0] > ucb[1]:
                group.append(0)
                X.append(x[0])
                outcome.append(x[0])
                x = x[1:]
            else:
                group.append(1)
                Y.append(y[0])
                outcome.append(y[0])
                y = y[1:]
    
    return group, outcome


if __name__ == '__main__':
    N = 2000 # Number of subjects that can be allocated in each group
    # analysis output, so we can read it directly to plot
    out_file = "comparison_pickle_2000.pkl"
    # Set seed for replicability
    np.random.seed(seed=123)
    # Create distributions
    control = np.random.normal(loc=0, scale=1, size=N)
    trt1 = np.random.normal(loc=0.1, scale=2, size=N)
    # Switches to turn on and off for various analysis
    run_simulations = True
    run_plots = True
    all_grps = True
    seperate = False
    agg_type = "avg"
    if run_simulations:
        # Now we call all the algorithms that can simulate various allocation
        # procedures
        # Calling vanilla A/B testing
        ab_group, ab_outcome = vanilla_AB(x=control, y=trt1)
        # calling ucb naive bandit algorithm
        ucb_bandit = Bandit(name='ucb_naive',
                            num_arms=2,
                            trt_dist_list=[control, trt1])
        ucb_bandit = ucb_naive(bandit=ucb_bandit, num_rounds=N,
                               num_arms=2)
        ucb_group, ucb_outcome = ucb_bandit.arm_tracker, ucb_bandit.reward_tracker
        # calling A/B testing with p value peeking
        peek_group, peek_outcome, peek_p = always_valid_p_value_emp(theta0=0,
                                                                    tau=1,
                                                                    x=control,
                                                                    y=trt1)




        # calling mixed algorithm that combines a/b and UCB
        mix_group, mix_outcome = mixed_alg(x=control, y=trt1, perc_ab=0.3)

        
        ab_con_lis, ab_trt_lis = split_outcome_lists(ab_group, ab_outcome)
        ucb_con_lis, ucb_trt_lis = split_outcome_lists(ucb_group, ucb_outcome)
        peek_con_lis, peek_trt_lis = split_outcome_lists(peek_group, peek_outcome)
        mix_con_lis, mix_trt_lis = split_outcome_lists(mix_group, mix_outcome)
        ab_lis = outcome_list(ab_outcome)
        ucb_lis = outcome_list(ucb_outcome)
        peek_lis = outcome_list(peek_outcome)
        mix_lis = outcome_list(mix_outcome)
        my_dict = {"ab_group": ab_group, "ab_outcome": ab_outcome, "ab_lis": ab_lis, "ab_con_lis": ab_con_lis,
                   "ab_trt_lis": ab_trt_lis,
                   "ucb_group": ucb_group, "ucb_outcome": ucb_outcome, "ucb_lis": ucb_lis, "ucb_con_lis": ucb_con_lis,
                   "ucb_trt_lis": ucb_trt_lis, "peek_group":peek_group, "peek_outcome":peek_outcome, "peek_lis":peek_lis,
                   "peek_con_lis":peek_con_lis, "peek_trt_lis":peek_trt_lis, "mix_group":mix_group,
                   "mix_outcome":mix_outcome, "mix_lis":mix_lis, "mix_con_lis":mix_con_lis, "mix_trt_lis":mix_trt_lis}

        # this is because, the above dictionary will be of unequal length
        df = pd.DataFrame.from_dict(my_dict, orient='index')
        df = df.transpose()
      # pickle.dump(df, open(out_file, 'wb'))

    if run_plots:
       # df = pickle.load(open(out_file, 'rb'))
        if all_grps:
            ab_m = stats(df.ab_lis, agg="avg")
            ucb_m = stats(df.ucb_lis, agg="avg")
            peek_m = stats(df.peek_lis, agg="avg")
            mix_m = stats(df.mix_lis, agg="avg")
            ab_v = stats(df.ab_lis, agg="var")
            ucb_v = stats(df.ucb_lis, agg="var")
            peek_v = stats(df.peek_lis, agg="var")
            mix_v = stats(df.mix_lis, agg="var")
            ab_r = stats(df.ab_lis, agg="rse", target=1)
            ucb_r = stats(df.ucb_lis, agg="rse", target=1)
            peek_r = stats(df.peek_lis, agg="rse", target=1)
            mix_r = stats(df.mix_lis, agg="rse", target=1)
            ab_s = stats(df.ab_lis, agg="sum")
            ucb_s = stats(df.ucb_lis, agg="sum")
            peek_s = stats(df.peek_lis, agg="sum")
            mix_s = stats(df.mix_lis, agg="sum")
            trans_dict = {"ab_m": ab_m, "ucb_m": ucb_m, "peek_m": peek_m,
                          "mix_m": mix_m, "ab_v": ab_v, "ucb_v": ucb_v,
                          "peek_v": peek_v, "mix_v": mix_v, "ab_r": ab_r,
                          "ucb_r": ucb_r, "peek_r": peek_r, "mix_r": mix_r,
                          "ab_s": ab_s, "ucb_s": ucb_s, "peek_s": peek_s,
                          "mix_s": mix_s}
    
            df2 = pd.DataFrame.from_dict(trans_dict, orient='index')
            df2 = df2.transpose()
            df2['time'] = df2.index
            df2.to_csv("analysis_overall.csv")

            ab_m = stats(df.ab_con_lis, agg="avg")
            ucb_m = stats(df.ucb_con_lis, agg="avg")
            peek_m = stats(df.peek_con_lis, agg="avg")
            mix_m = stats(df.mix_con_lis, agg="avg")
            ab_v = stats(df.ab_con_lis, agg="var")
            ucb_v = stats(df.ucb_con_lis, agg="var")
            peek_v = stats(df.peek_con_lis, agg="var")
            mix_v = stats(df.mix_con_lis, agg="var")
            ab_r = stats(df.ab_con_lis, agg="rse", target=1)
            ucb_r = stats(df.ucb_con_lis, agg="rse", target=1)
            peek_r = stats(df.peek_con_lis, agg="rse", target=1)
            mix_r = stats(df.mix_con_lis, agg="rse", target=1)
            ab_s = stats(df.ab_con_lis, agg="sum")
            ucb_s = stats(df.ucb_con_lis, agg="sum")
            peek_s = stats(df.peek_con_lis, agg="sum")
            mix_s = stats(df.mix_con_lis, agg="sum")
    
            trans_dict = {"ab_m": ab_m, "ucb_m": ucb_m, "peek_m": peek_m,
                          "mix_m": mix_m, "ab_v": ab_v, "ucb_v": ucb_v,
                          "peek_v": peek_v, "mix_v": mix_v, "ab_r": ab_r,
                          "ucb_r": ucb_r, "peek_r": peek_r, "mix_r": mix_r,
                          "ab_s": ab_s, "ucb_s": ucb_s, "peek_s": peek_s,
                          "mix_s": mix_s}
    
            df2 = pd.DataFrame.from_dict(trans_dict, orient='index')
            df2 = df2.transpose()
            df2['time'] = df2.index
            df2.to_csv("analysis_rontrol.csv")
    
            ab_m = stats(df.ab_trt_lis, agg="avg")
            ucb_m = stats(df.ucb_trt_lis, agg="avg")
            peek_m = stats(df.peek_trt_lis, agg="avg")
            mix_m = stats(df.mix_trt_lis, agg="avg")
            ab_v = stats(df.ab_trt_lis, agg="var")
            ucb_v = stats(df.ucb_trt_lis, agg="var")
            peek_v = stats(df.peek_trt_lis, agg="var")
            mix_v = stats(df.mix_trt_lis, agg="var")
            ab_r = stats(df.ab_trt_lis, agg="rse", target=2)
            ucb_r = stats(df.ucb_trt_lis, agg="rse", target=2)
            peek_r = stats(df.peek_trt_lis, agg="rse", target=2)
            mix_r = stats(df.mix_trt_lis, agg="rse", target=2)
            ab_s = stats(df.ab_trt_lis, agg="sum")
            ucb_s = stats(df.ucb_trt_lis, agg="sum")
            peek_s = stats(df.peek_trt_lis, agg="sum")
            mix_s = stats(df.mix_trt_lis, agg="sum")
            trans_dict = {"ab_m": ab_m, "ucb_m": ucb_m, "peek_m": peek_m,
                          "mix_m": mix_m, "ab_v": ab_v, "ucb_v": ucb_v,
                          "peek_v": peek_v, "mix_v": mix_v, "ab_r": ab_r,
                          "ucb_r": ucb_r, "peek_r": peek_r, "mix_r": mix_r,
                          "ab_s": ab_s, "ucb_s": ucb_s, "peek_s": peek_s,
                          "mix_s": mix_s}
    
            df2 = pd.DataFrame.from_dict(trans_dict, orient='index')
            df2 = df2.transpose()
            df2['time'] = df2.index
            df2.to_csv("analysis_trt.csv")
    
            print("Done Running")