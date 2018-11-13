library(dplyr)
library(ggplot2)
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


donor <- donor %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() %>% 
  group_by(alpha, cumsum_cost) %>% n()

don_opt <- don_opt %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() 

referral <- referral %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() 

ref_opt <- ref_opt %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(alpha, ite, cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward)) %>% ungroup() 


ggplot(donor, aes(cumsum_cost, cumsum_reward)) + geom_line(color="blue") + geom_line(data=don_opt, aes(cumsum_cost, cumsum_reward), color="red")

ggplot(referral, aes(cumsum_cost, cumsum_reward)) + geom_line(color="blue") + geom_line(data=ref_opt, aes(cumsum_cost, cumsum_reward), color="red")
