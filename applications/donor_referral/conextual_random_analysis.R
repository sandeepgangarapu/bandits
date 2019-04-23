library(dplyr)
library(ggplot2)
library(knitr)
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\algorithms")

donor <- read.csv('donor_linucb.csv') %>% group_by(alpha, ite) %>%  mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward)) %>% ungroup() 
referral <- read.csv('referral_linucb.csv') %>% group_by(alpha, ite) %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward)) %>% ungroup() 


# Filtering into new df for contexts that were allocated optimally

don_opt <- donor %>% filter(optimal==1)  %>% group_by(alpha, ite) %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward)) %>% ungroup() 
ref_opt <- referral %>% filter(optimal==1)  %>% group_by(alpha, ite) %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward)) %>% ungroup() 


donor <- donor %>% select(alpha, ite, cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() %>% 
  group_by(alpha, cumsum_cost) %>% summarize(mean_reward=mean(cumsum_reward), se_reward = sd(cumsum_reward)/sqrt(n()))

don_opt <- don_opt %>% select(alpha, ite, cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() %>% 
  group_by(alpha, cumsum_cost) %>% summarize(mean_reward=mean(cumsum_reward), se_reward = sd(cumsum_reward)/sqrt(n()))

referral <- referral %>% select(alpha, ite, cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() %>% 
  group_by(alpha, cumsum_cost) %>% summarize(mean_reward=mean(cumsum_reward), se_reward = sd(cumsum_reward)/sqrt(n()))

ref_opt <- ref_opt %>% select(alpha, ite, cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() %>% 
  group_by(alpha, cumsum_cost) %>% summarize(mean_reward=mean(cumsum_reward), se_reward = sd(cumsum_reward)/sqrt(n()))


  
  ggplot(donor %>% filter(alpha==1), aes(cumsum_cost, mean_reward)) +
    geom_line(color='red') +
    geom_line(data=don_opt %>% filter(alpha==1& ite=1), aes(cumsum_cost, mean_reward), color='blue') +
    ggsave("donor_linucb_graph.png", width = 8, height = 5)
  
  ggplot(donor %>% filter(alpha==15& ite=1), aes(cumsum_cost, mean_reward)) +
    geom_line(color='red') +
    geom_line(data=don_opt %>% filter(alpha==15), aes(cumsum_cost, mean_reward), color='blue') +
    ggsave("donor_linucb_graph.png", width = 8, height = 5)
  
  