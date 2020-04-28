setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\ucb_inf")
library(ggplot2)
library(dplyr)
library(tidyr)

group_outcome <- read.csv("group_outcome_sim.csv") %>% 
  group_by(ite,alg) %>% mutate(x=row_number()) %>%
  ungroup()

regret_mse <- read.csv("regret_mse.csv") %>% 
  group_by(ite,alg) %>% mutate(x=row_number()) %>% ungroup()  %>%
  ungroup() 

# Group allocation graph

df <- group_outcome %>% filter(ite==0) 

df_regret <- regret_mse %>% filter(ite==0)

ggplot(df) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(alg ~.) + theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = 6)) +   theme(axis.text.y = element_text(size = rel(1))) 
# +   ggsave("group_all.png", width = 10, height = 6, dpi=300, units="in")


ggplot(df_regret) + geom_line(aes(x=x, y=regret, color=alg)) +
  ggsave("regret_eps_n.png", width = 10, height = 6, dpi=300, units="in")


ggplot(df_regret) + geom_line(aes(x=x, y=mean_mse, color=alg))

ggplot(df_regret) + geom_line(aes(x=x, y=var_mse, color=alg))


# results for chi simulation for ucb-inf

chi_group <- read.csv("group_eps_sim.csv")

chi_group <- chi_group %>% mutate(chi = round(chi,2)) %>%
  group_by(chi) %>% mutate(x=row_number()) %>% ungroup()


ggplot(chi_group) +
  geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(chi ~.) + theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = 6)) + 
  theme(axis.text.y = element_text(size = rel(1))) 

chi_regret <- read.csv("regret_mse_eps_sim.csv") %>% 
  mutate(chi = factor(round(chi,2))) %>% group_by(chi) %>% 
  mutate(x=row_number())

ggplot(chi_regret) + 
  geom_line(aes(x=x, y=regret, color=chi)) + theme_bw()

ggplot(chi_regret %>% filter(x<500)) + 
  geom_line(aes(x=x, y=mean_mse, color=chi))


# Small sample properties of algortihms

means = c(2.9013279, 1.01096483, 2.20275192, 0.25, 3.87932978, 3.44364749,
          0.4534811, 1.86620435, 4.40527515, 3)

vars = c(4.57899815, 2.90258678, 4.77255198, 1.56356794, 4.65556446, 0.89617444,
         3.26568194, 3.43573849, 1, 4.83549103)


# calculate bias

# Best arm of only AB and UCB
best <- group_outcome %>%  filter(group==which.max(means)-1 & type!='UCB-INF') %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% 
  mutate(bias = mn-max(means)) %>% group_by(type) %>%
  summarise(Bias = mean(bias), sd = sd(bias)/sqrt(n()))


ggplot(best, aes(x=type, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*sd, ymax=Bias+qnorm(0.975)*sd), width=0.2) +
  theme_bw()  + ylim(-1, 1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("best_bias_2.png", width = 5, height = 5, scale = 0.7, units="in")

# Best arm of AB, UCB and UCB-INF


best_all <- group_outcome %>%  filter(group==which.max(means)-1) %>% group_by(type, ite) %>%
  summarise(mn = mean(outcome)) %>% 
  mutate(bias = mn-max(means)) %>% group_by(type) %>%
  summarise(Bias = mean(bias), sd = sd(bias)/sqrt(n()))

ggplot(best_all, aes(x=type, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*sd, ymax=Bias+qnorm(0.975)*sd), width=0.2) +
  theme_bw()  + ylim(-1, 1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("best_bias_all.png", width = 5, height = 5, scale = 0.7, units="in")



# Worst arm of only AB and UCB


worst <- group_outcome %>%  filter(group==which.min(means)-1 & type!='UCB-INF')  %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% 
  mutate(bias = mn-min(means)) %>% group_by(type) %>%
  summarise(Bias = mean(bias), sd = sd(bias)/sqrt(n()))


ggplot(worst, aes(x=type, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*sd, ymax=Bias+qnorm(0.975)*sd), width=0.2) +
  theme_bw()  + ylim(-1, 1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("worst_bias_2.png", width = 5, height = 5, scale = 0.7, units="in")


# Worst  arm of AB, UCB and UCB-INF


worst_all <- group_outcome %>%  filter(group==which.min(means)-1)  %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% 
  mutate(bias = mn-min(means)) %>% group_by(type) %>%
  summarise(Bias = mean(bias), sd = sd(bias)/sqrt(n()))


ggplot(worst_all, aes(x=type, y=Bias)) + geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=Bias-qnorm(0.975)*sd, ymax=Bias+qnorm(0.975)*sd), width=0.2) +
  theme_bw()  + ylim(-1, 1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("worst_bias_all.png", width = 5, height = 5, scale = 0.7, units="in")


# mse of best estimate

# Best arm of only AB and UCB
best <- group_outcome %>%  filter(group==which.max(means)-1 & type!='UCB-INF') %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% group_by(type) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-max(means))^2+v)


ggplot(best, aes(x=type, y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-0.1, 0.1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("best_mse_2.png", width = 5, height = 5, scale = 0.7, units="in")

# Best arm of AB, UCB and UCB-INF

best <- group_outcome %>%  filter(group==which.max(means)-1) %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% group_by(type) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-max(means))^2+v)


ggplot(best, aes(x=type, y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-0.1, 0.1) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("best_mse_all.png", width = 5, height = 5, scale = 0.7, units="in")

# worst arm of only AB and UCB
worst <- group_outcome %>%  filter(group==which.min(means)-1 & type!='UCB-INF') %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% group_by(type) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-min(means))^2+v)


ggplot(worst, aes(x=type, y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-1.5, 1.5) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("worst_mse_2.png", width = 5, height = 5, scale = 0.7, units="in")

# Best arm of AB, UCB and UCB-INF
worst <- group_outcome %>%  filter(group==which.min(means)-1) %>%
  group_by(type, ite) %>% summarise(mn = mean(outcome)) %>% group_by(type) %>%
  summarise(m=mean(mn), v=var(mn)) %>% mutate(mse=(m-min(means))^2+v)


ggplot(worst, aes(x=type, y=mse)) + geom_bar(stat="identity") +
  theme_bw()  + ylim(-1.5, 1.5) + theme(axis.title.x=element_blank()) +
  geom_hline(yintercept = 0) +
  ggsave("worst_mse_all.png", width = 5, height = 5, scale = 0.7, units="in")



# estimate normality sim
sd <- read.csv("group_outcome_sim.csv")

library(rcompanion)
worst_ab <- sd %>%  filter(group==which.min(means)-1 & type=='ab') %>%
  group_by(ite) %>% summarise(mn = mean(outcome)) 

plotNormalHistogram(worst_ab$mn, prob = FALSE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 1000)



worst_ucb <- sd %>%  filter(group==which.min(means)-1 & type=='ucb') %>%
  group_by(ite) %>% summarise(mn = mean(outcome)) 

plotNormalHistogram(worst_ucb$mn, prob = FALSE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 100)




worst_ucb_inf <- sd %>%  filter(group==which.min(means)-1 & type=='ucb_inf_var_prop') %>%
  group_by(ite) %>% summarise(mn = mean(outcome)) 

plotNormalHistogram(worst_ucb$mn, prob = FALSE, col = "gray", main = "",
                    linecol = "blue", lwd = 2, length = 100)


shapiro.test(worst_ab$mn)
shapiro.test(worst_ucb$mn)
shapiro.test(worst_ucb_inf$mn)

