from bandits.bandit_comparison_simulation import BanditSimulation
import numpy as np

true_means = [3.36139279, 2.440392, 4.12747587, 0.25, 4.04024982, 2.8280871, 1.48811249, 0.2334786, 4.953137, 3]
true_vars = [3.84896514, 3.7338355, 1.88719468, 2.47073726, 4.64474196, 1.97727022, 4.86978148, 2.62207358, 1, 4.06654206]
arm_means=np.random.uniform(0, 5, 5)
arm_vars=np.random.uniform(0, 5, 5)

sim = BanditSimulation(seed=10, num_ite=2, arm_means=true_means,
                       arm_vars=true_vars,
                       eps_inf=0.5,
                       horizon=100,
                       alg_list=['ab', 'ucb', 'eps_greedy', 'ucb_inf_eps', 'thomp', 'thomp_inf_eps'],
                       output_file_path=None)
output, prop = sim.run_simulation()
print(prop.head)
