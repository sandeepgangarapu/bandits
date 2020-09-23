setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")
library(dplyr)
library(tidyr)

df <- read.csv("test_weighed.csv") %>% drop_na()

df_sub <- df %>% mutate(bias = true_mean-mn) %>% group_by(main_ite, alg, grp, weight_lis) %>%
  summarise(mn_bias = mean(bias), sd_bias = sd(bias),
            upr = mn_bias + qnorm(0.975)*sd_bias, lwr = mn_bias - qnorm(0.975)*sd_bias,
            upr_greater = upr>0, lwr_lesser = lwr<0,
            diff_sign = !xor(upr_greater, lwr_lesser)) %>% ungroup()


df_sign <- df_sub %>% group_by(alg, diff_sign) %>% summarise(cnt = n(), diff_val_mn = mean(upr-lwr)) %>% ungroup()

detective_df <- df_sub %>% group_by(alg, diff_sign) %>% summarise(avg_weight = mean(weight_lis))


normality_df <- df %>% group_by(alg, main_ite, grp) %>%
  summarise(test_pass = shapiro.test(mn)$p.value>0.05) %>% group_by(alg, test_pass) %>% 
  summarise(cnt = n())

print(normality_df)