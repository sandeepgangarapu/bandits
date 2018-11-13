library(dplyr)
library(ggplot2)
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\algorithms")

donor <- read.csv('donor_linucb.csv') %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward))
referral <- read.csv('referral_linucb.csv') %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward))


# Filtering into new df for contexts that were allocated optimally

don_opt <- donor %>% filter(optimal==1) %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward))
ref_opt <- referral %>% filter(optimal==1) %>% mutate(cumsum_cost = cumsum(cost)) %>%
  mutate(cumsum_reward = cumsum(reward))


donor <- donor %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward))

don_opt <- don_opt %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward))

referral <- referral %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward))

ref_opt <- ref_opt %>% select(cumsum_cost, cumsum_reward) %>%
  group_by(cumsum_cost) %>% summarise(cumsum_reward=max(cumsum_reward))


ggplot(donor, aes(cumsum_cost, cumsum_reward)) + geom_line(color="blue") + geom_line(data=don_opt, aes(cumsum_cost, cumsum_reward), color="red")

ggplot(referral, aes(cumsum_cost, cumsum_reward)) + geom_line(color="blue") + geom_line(data=ref_opt, aes(cumsum_cost, cumsum_reward), color="red")
