suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(dplyr)
    library(tidyr)
  })


setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


data <- read.csv("regret_3_2000_1.csv")
#data <- read.csv("bias_1_2000.csv")


seed_algs <- c("ab", "ucb", "thomp", "eps_greedy")
inf_algs <- c("ucb_inf_eps", "thomp_inf_eps")
est_algs <- c("thomp_ipw", "thomp_aipw", "thomp_inf_eps_ipw", "thomp_inf_eps_aipw", "thomp_eval_aipw", "thomp_inf_eps_eval_aipw")
thomp_algs = c( "thomp", "thomp_inf_eps")
ucb_algs <- c("ucb_inf_eps", "ucb")
adv_algs <- c(thomp_algs, ucb_algs)


group_outcome <- data %>% filter(alg %in% c(seed_algs, inf_algs)) %>%
  select(alg, group, ite, outcome) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>% filter(x<500) %>% 
  ungroup()

regret_mse <- data %>% filter(alg %in% c(seed_algs, inf_algs)) %>%
  select(alg, regret, ite, mean_mse, var_mse) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>% filter(x<500) %>% 
  ungroup()



xlimit <- max(group_outcome$x) + 500


# Setting new WD so that results are saved in different folder
setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\results")



df_regret <- regret_mse %>% filter(ite==0)

ggplot(df_regret, aes(x=x, y=regret)) + geom_line(aes(color=alg)) +
  labs(title = "Regret") + xlim(0,xlimit) +
  theme_bw() 


df_mse <- regret_mse %>% select(-c(regret, var_mse)) %>% rename(mse = mean_mse) 

# %>% 
#   group_by(x, alg) %>%
#   summarise(mn_mse = mean(mse), se_mse = sd(mse)/sqrt(n()))


ggplot(df_mse %>% filter(ite==0),aes(x=x, y=mse)) +
geom_line(aes(color=alg)) + 
#geom_ribbon(aes(ymin=mn_mse-(1.96*se_mse), ymax=mn_mse+(1.96*se_mse),group=alg), alpha=0.4,  fill="grey70") +
labs(title = "MSE of Mean") +  xlim(0,xlimit) + 
 theme_bw() 
