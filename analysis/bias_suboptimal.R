suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(dplyr)
    library(tidyr)
  })


setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")

true_means = c(1.5,2.8,3)

true_vars = c(1,1,1)

grp = c(0:(length(true_means)-1))
true_df <- data.frame(grp, true_means, true_vars)
true_df <- true_df %>% mutate(grp=factor(grp))

seed_algs <- c("ab", "ucb", "thomp", "eps_greedy")
inf_algs <- c("ucb_inf_eps", "thomp_inf_eps")
est_algs <- c("thomp_ipw", "thomp_aipw", "thomp_inf_eps_ipw",
              "thomp_inf_eps_aipw", "thomp_eval_aipw", 'thomp_inf_eps_eval_aipw',
              "ucb_aipw", "ucb_ipw",  "ucb_inf_eps_aipw", "ucb_inf_eps_ipw" )
thomp_algs <- c( "thomp", "thomp_inf_eps")
thomp_est_algs <- c( "thomp_ipw", "thomp_aipw", "thomp_eval_aipw", "thomp_inf_eps_ipw",
                     "thomp_inf_eps_aipw", 'thomp_inf_eps_eval_aipw')
ucb_algs <- c("ucb_inf_eps", "ucb")
ucb_est_algs <- c("ucb_aipw", "ucb_ipw",  "ucb_inf_eps_aipw", "ucb_inf_eps_ipw")
adv_algs <- c(thomp_algs, ucb_algs)
#data <- read.csv("bias_500_50.csv")
#data <- read.csv("bias_100_500.csv")
data <- read.csv("res_40_500.csv")
setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\results")


group_outcome <- data  %>%  filter(alg %in% c(seed_algs, inf_algs)) %>%
  select(alg, group, ite, outcome) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>%
  ungroup()
means <- group_outcome  %>% group_by(group, alg, ite) %>%
  summarise(mn = mean(outcome)) %>% ungroup()


weighed_means <- data %>%  filter(alg %in% est_algs) %>%
  select(alg, ite, group, mean_est) %>% 
  group_by(ite,alg, group) %>%
  mutate(x=row_number(), y= max(row_number())) %>%
  ungroup() %>% filter(x==y) %>% select(-c(x,y)) %>% rename(mn = mean_est)


best <- means %>% filter(group==which.max(true_means)-1) %>% 
  mutate(bias = mn-max(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- 0.3

ggplot(best %>% filter(alg %in% c('thomp', 'ab', 'thomp_inf_eps')), aes(x=alg, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  # labs(title = "Bias of best arm") +
  # theme(axis.text.x = element_text(angle = 90)) +
  ggsave("bias_best_1.png", width = 4, height = 4, dpi=300, units="in")

best <- means %>% filter(group==which.max(true_means)-1) %>%
  group_by(alg) %>%
  summarise(mse = mean((mn-max(true_means))^2))
  #summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-max(true_means))^2+v)


ggplot(best  %>% filter(alg %in% c('thomp', 'ab', 'thomp_inf_eps')), aes(x=alg, y=mse)) + 
  geom_bar(stat="identity") +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  theme(axis.title.x=element_blank())  +
  ggsave("mse_best_1.png", width = 4, height = 4, dpi=300, units="in")


worst <- means %>% filter(group==which.min(true_means)-1) %>% 
  mutate(bias = mn-min(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))


ggplot(worst %>% filter(alg %in% c('thomp', 'ab', 'thomp_inf_eps')), aes(x=alg, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  # labs(title = "Bias of Worst arm") +
  # theme(axis.text.x = element_text(angle = 90)) +
  ggsave("bias_worst_1.png", width = 4, height = 4, dpi=300, units="in")

worst <- means %>%  filter(group==which.min(true_means)-1)  %>% group_by(alg) %>%
  summarise(mse = mean((mn-min(true_means))^2))
  #summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-min(true_means))^2)


ggplot(worst %>% filter(alg %in% c('thomp', 'ab', 'thomp_inf_eps')), aes(x=alg, y=mse)) +
  geom_bar(stat="identity") +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  # labs(title = "MSE of Worst arm") +
  theme(axis.title.x=element_blank()) +
  ggsave("mse_worst_1.png", width = 4, height = 4, dpi=300, units="in")


library(rcompanion)
worst_ab <- means %>%  filter(group==which.min(true_means)-1 & alg=='ab')

plotNormalHistogram(worst_ab$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)


shapiro.test(worst_ab$mn)


best_ab <- means %>%  filter(group==which.max(true_means)-1 & alg=='ab')

plotNormalHistogram(best_ab$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)

shapiro.test(best_ab$mn)



worst_thomp <- means %>%  filter(group==which.min(true_means)-1 & alg=='thomp') 

plotNormalHistogram(worst_thomp$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=2,  col="red")


shapiro.test(worst_thomp$mn)


best_thomp <- means %>%  filter(group==which.max(true_means)-1 & alg=='thomp') 

plotNormalHistogram(best_thomp$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=3,  col="red")

shapiro.test(best_thomp$mn)



worst_thomp_inf_eps <- means %>%  filter(group==which.min(true_means)-1 & alg=='thomp_inf_eps') 

plotNormalHistogram(worst_thomp_inf_eps$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=2,  col="red")


shapiro.test(worst_thomp_inf_eps$mn)


best_thomp_inf_eps <- means %>%  filter(group==which.max(true_means)-1 & alg=='thomp_inf_eps') 

plotNormalHistogram(best_thomp_inf_eps$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=3,  col="red")

shapiro.test(best_thomp_inf_eps$mn)


# weighed algorithms


best <- weighed_means %>% filter(alg %in% c('thomp_ipw', 'thomp_aipw')) %>% filter(group==which.max(true_means)-1) %>% 
  mutate(bias = mn-max(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))

ylimit <- max(abs(c(best$Bias + qnorm(0.975)*best$se, best$Bias - qnorm(0.975)*best$se)))
ylimit <- 0.3

ggplot(best, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) + labs(title = "Bias of best arm") +
  theme(axis.text.x = element_text(angle = 90)) 


worst_1 <- weighed_means %>% filter(alg %in% c('thomp_inf_eps_ipw', 'thomp_inf_eps_aipw')) %>% filter(group==which.min(true_means)-1) %>% 
  mutate(bias = mn-min(true_means)) %>% group_by(alg) %>%
  summarise(Bias = mean(bias), se = sd(bias)/sqrt(n()))
worst <- rbind(worst, worst_1)

ylimit <- max(abs(c(worst$Bias + qnorm(0.975)*worst$se, worst$Bias - qnorm(0.975)*worst$se)))
ylimit <- 10

ggplot(worst, aes(x=reorder(alg, abs(Bias)), y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*se, ymax=Bias+qnorm(0.975)*se), width=0.2) +
  theme_bw()  + ylim(-ylimit, ylimit) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  theme(axis.text.x = element_text(angle = 90)) +
  ggsave("bias_worst_weighed.png", width = 7, height = 4, dpi=300, units="in")



best_thomp_ipw <- weighed_means %>%  filter(group==which.max(true_means)-1 & alg=='thomp_ipw') 

plotNormalHistogram(best_thomp_ipw$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=3,  col="red")

shapiro.test(best_thomp_ipw$mn)


worst_thomp_ipw <- weighed_means %>%  filter(group==which.min(true_means)-1 & alg=='thomp_ipw') 

plotNormalHistogram(worst_thomp_ipw$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=2,  col="red")

shapiro.test(worst_thomp_ipw$mn)


best_thomp_aipw <- weighed_means %>%  filter(group==which.max(true_means)-1 & alg=='thomp_aipw') 

plotNormalHistogram(best_thomp_aipw$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=3,  col="red")

shapiro.test(best_thomp_aipw$mn)


worst_thomp_aipw <- weighed_means %>%  filter(group==which.min(true_means)-1 & alg=='thomp_aipw') 

plotNormalHistogram(worst_thomp_aipw$mn, prob = TRUE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000, breaks= 50)
abline(v=2,  col="red")

shapiro.test(worst_thomp_aipw$mn)
