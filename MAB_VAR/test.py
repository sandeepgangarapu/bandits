from bandits.bandit import Bandit
from bandits.algorithms.ucb_naive import ucb_naive
from bandits.MAB_VAR.stats import rmse_outcome

group = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]
outcome = [7,3,5,0,2,8,3,7,5,8,2,3,5,2,7,2,3]

a = [8.628506,10.310681,10.761258,10.135495,10.779007,12.805618,11.503457,9.652288,11.239432,9.442110,10.824988,10.140028,11.919773,10.956450,12.206700,9.830582,10.733304,10.461339,10.627789,11.241039]
b = [13.828054,10.078001,7.459152,8.086171,11.020271,13.056955,13.075182,15.647358,9.581052,12.757539,17.829993,11.841114,14.626951,9.911239,10.866524,11.222787,12.702886,9.073667,11.526477,9.672839]
c = [10.326639,7.685557,8.322819,5.545426,7.717036,8.598726,12.854831, 10.115260,6.270734,14.239394,9.351070, 10.862573,6.803008,7.093309,7.347557,9.186338,10.136632,16.146189,12.214350,6.502766]
num_arms = 3


out = [a,c,b]

ucb_bandit = Bandit(name='ucb_naive', num_arms=num_arms,
                    trt_dist_list=out)

ucb_bandit = ucb_naive(bandit=ucb_bandit, num_rounds=20, num_arms=num_arms)


print(ucb_bandit.total_reward_tracker)