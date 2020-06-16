# this is to check if ipw and aipw estimates are unbiased for both best and worst arms

library(dplyr)
library(tidyr)
library(ggplot2)

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")

true_means = c(0.25, 1.82, 1.48, 2.25, 2)

true_vars = c(2.84,  1.97, 2.62, 1, 2.06)

grp = c(0:(length(true_means)-1))
true_df <- data.frame(grp, true_means, true_vars)
true_df <- true_df %>% mutate(grp=factor(grp))


data <- read.csv("sim_thomp_ucb_300_1000.csv")


seed_algs <- c("ab", "ucb", "thomp", "eps_greedy")
inf_algs <- c("ucb_inf_eps", "thomp_inf_eps")
est_algs <- c("thomp_ipw", "thomp_aipw", "thomp_inf_eps_ipw",
              "thomp_inf_eps_aipw", "thomp_eval_aipw", "thomp_inf_eps_eval_aipw", 
              "ucb_aipw", "ucb_ipw",  "ucb_inf_eps_aipw", "ucb_inf_eps_ipw", 
              "ucb_eval_aipw", "ucb_inf_eps_eval_aipw" )
thomp_algs <- c( "thomp", "thomp_inf_eps")
thomp_est_algs <- c( "thomp_ipw", "thomp_aipw", "thomp_inf_eps_ipw",
                "thomp_inf_eps_aipw")
ucb_algs <- c("ucb_inf_eps", "ucb")
adv_algs <- c(thomp_algs, ucb_algs)


group_outcome <- data %>% filter(alg %in% thomp_algs) %>%
  select(alg, group, ite, outcome) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>%
  ungroup()


weighed_means <- data %>% filter(alg %in% thomp_est_algs) %>% 
  select(alg, ite, group, mean_est) %>% 
  group_by(ite,alg, group) %>%
  mutate(x=row_number(), y= max(row_number())) %>%
  ungroup() %>% filter(x==y) %>% select(-c(x,y)) %>% rename(mn = mean_est)

 means <- group_outcome  %>% group_by(group, alg, ite) %>%
  summarise(mn = mean(outcome)) %>% ungroup()


all_means <- rbind(means, weighed_means)


xlimit <- max(group_outcome$x) + 1000


# Setting new WD so that results are saved in different folder
setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\results")

# Bias of the Best arm


best <- all_means %>% filter(group==which.max(true_means)-1) %>% 
  mutate(bias = mn-max(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(best$Bias + qnorm(0.975)*best$se, best$Bias - qnorm(0.975)*best$se)))
ylimit <- 1.5*ylimit

ggplot(best, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of best arm") +
  theme(axis.text.x = element_text(angle = 90)) 


# Bias of the worst arm

worst <- all_means %>% filter(group==which.min(true_means)-1) %>% 
  mutate(bias = mn-min(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(worst$Bias + qnorm(0.975)*worst$se, worst$Bias - qnorm(0.975)*worst$se)))
ylimit <- 1.5*ylimit

ggplot(worst, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of Worst arm") + theme(axis.text.x = element_text(angle = 90)) 



## MSE of Best arm 


best <- all_means %>%  filter(group==which.max(true_means)-1)  %>% group_by(alg) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-max(true_means))^2+v)


ylimit <- max(best$mse)
ylimit <- 1.5*ylimit

ggplot(best, aes(x=reorder(alg, mse), y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "MSE of best arm") + 
  theme(axis.title.x=element_blank()) + theme(axis.text.x = element_text(angle = 90)) 


## MSE of Worst arm 


worst <- all_means %>%  filter(group==which.min(true_means)-1)  %>% group_by(alg) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-min(true_means))^2)

ylimit <- max(worst$mse)
ylimit <- 1.5*ylimit

ggplot(worst, aes(x=reorder(alg, mse), y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "MSE of Worst arm") + 
  theme(axis.title.x=element_blank()) + theme(axis.text.x = element_text(angle = 90)) 



# this analysis is for seeing how the estimate behaves asymptotically
# x axis - horizon
# y axis - estimate value


best_weighed_means <- data %>% filter(alg %in% est_algs) %>% 
  select(alg, ite, group, mean_est) %>%  group_by(ite, alg) %>%
  mutate(x=row_number()) %>%
  ungroup() %>% filter(ite==0 & group == which.max(true_means)-1)

ggplot(best_weighed_means, aes(x=x, y=mean_est)) + geom_line() + facet_wrap(~alg, nrow = 3) +
  theme_bw() + theme(axis.title.x=element_blank()) + geom_hline(yintercept= max(true_means)) +
  geom_hline(yintercept = 0) + labs(title = "Mean estimate of best arm") + 
  theme(axis.title.x=element_blank()) + theme(axis.text.x = element_text(angle = 90)) 



worst_weighed_means <- data %>% filter(alg %in% est_algs) %>% 
  select(alg, ite, group, mean_est) %>%  group_by(ite, alg) %>%
  mutate(x=row_number()) %>%
  ungroup() %>% filter(ite==0 & group == which.min(true_means)-1)

ggplot(worst_weighed_means, aes(x=x, y=mean_est)) + geom_line() + facet_wrap(~alg, nrow = 3) +
  theme_bw() + theme(axis.title.x=element_blank()) + geom_hline(yintercept= min(true_means)) +
  geom_hline(yintercept = 0) + labs(title = "Mean estimate of worst arm") + 
  theme(axis.title.x=element_blank()) + theme(axis.text.x = element_text(angle = 90)) 




# The below analysis is for ucb based algorithms with
# propensity calculated in three different ways

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")



data <- read.csv("sim_ucb_ipw_aipw_3.csv")

true_means = c(0.25, 2.25, 2)
true_vars = c(2.84, 1, 2.06)

true_means = c(1,2,3)
true_vars = c(1,1,1)

grp = c(0:(length(true_means)-1))
true_df <- data.frame(grp, true_means, true_vars)
true_df <- true_df %>% mutate(grp=factor(grp))

weighed_means <- data %>% 
  group_by(ite, group) %>%
  mutate(x=row_number(), y= max(row_number())) %>%
  ungroup() %>% filter(x==y) %>% select(-c(x,y, outcome)) %>% 
  pivot_longer(-c(group, ite), names_to='alg', values_to='mn')


# Bias of the best arm


best <- weighed_means %>% filter(group==which.max(true_means)-1) %>% 
  mutate(bias = mn-max(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(best$Bias + qnorm(0.975)*best$se, best$Bias - qnorm(0.975)*best$se)))
ylimit <- 1.5*ylimit

ggplot(best, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of best arm") +
  theme(axis.text.x = element_text(angle = 90)) 


# Bias of the worst arm

worst <- weighed_means %>% filter(group==which.min(true_means)-1) %>% 
  mutate(bias = mn-min(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(worst$Bias + qnorm(0.975)*worst$se, worst$Bias - qnorm(0.975)*worst$se)))
ylimit <- 1.5*ylimit

ggplot(worst, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of worst arm") +
  theme(axis.text.x = element_text(angle = 90)) 




worst <- weighed_means %>% filter(group==2) %>% 
  mutate(bias = mn-2) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(worst$Bias + qnorm(0.975)*worst$se, worst$Bias - qnorm(0.975)*worst$se)))
ylimit <- 1.5*ylimit

ggplot(worst, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of worst arm") +
  theme(axis.text.x = element_text(angle = 90)) 



