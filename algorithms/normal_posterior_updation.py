# This code is to test thompson sampling
# 2 sep - its working

import numpy as np
from math import sqrt

alpha_prior = 0.5
beta_prior = 0.5
mu_0_prior = 0
n_0_prior = 2
prior = [mu_0_prior, n_0_prior, alpha_prior, beta_prior]
for i in range(100):
    # tau_prior = np.random.gamma(alpha_prior, beta_prior)
    # sigma_mu_prior = sqrt(1 / (n_0_prior * tau_prior))
    # mu_prior = np.random.normal(mu_0_prior, sigma_mu_prior)
    x = np.random.normal(2, 2)
    print(x)
    n_0_post = n_0_prior + 1
    alpha_post = alpha_prior + 0.5
    beta_post = beta_prior + ((n_0_prior)/(n_0_post))*(((x-mu_0_prior)**2)/2)
    mu_0_post = (((n_0_prior)*(mu_0_prior)) + x)/(n_0_post)
    mean_gamma = alpha_post/beta_post
    print(mu_0_prior, n_0_prior, alpha_prior, beta_prior)
    mu_0_prior, n_0_prior, alpha_prior, beta_prior = mu_0_post, n_0_post, alpha_post, beta_post
    print("mean = ", mu_0_post, "std= ", sqrt(1/mean_gamma))