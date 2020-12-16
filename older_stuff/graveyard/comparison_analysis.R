library(dplyr)
library(ggplot2)
library(tidyr)

setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\peeking")
df <- read.csv("analysis_overall.csv") %>%  select(-starts_with("peek"))

df_m <- df %>% select(time, ab_m, ucb_m, mix_m) %>% gather("type", "mean", 2:4) 


df_v <- df %>% select(time, ab_v, ucb_v, mix_v) %>% gather("type", "variance", 2:4)

df_r <- df %>% select(time, ab_r, ucb_r, mix_r) %>% gather("type", "rse", 2:4)

df_s <- df %>% filter(time<2000) %>%  select(time, ab_s, ucb_s, mix_s) %>%
  rename(`A/B Testing`=ab_s, `UCB`=ucb_s, `Hybrid_alg`=mix_s) %>%
  gather("Algorithm", "Total Outcome", 2:4)

ggplot(df_m, aes(time, mean)) + geom_line(aes(colour=type))


ggplot(df_v, aes(time, variance)) + geom_line(aes(colour=type))

ggplot(df_s, aes(time, `Total Outcome`)) + geom_line(aes(colour=Algorithm)) +
  theme_bw() + ggsave('out.png', width = 8, height = 5)


df2 <- read.csv("analysis_rontrol.csv") 

df2_m <- df2 %>% select(time, ab_m, ucb_m, mix_m) %>% gather("type", "mean", 2:4) 


df2_v <- df2 %>% select(time, ab_v, ucb_v, mix_v) %>%
  rename(`A/B Testing`=ab_v, `UCB`=ucb_v, `Hybrid_alg`=mix_v) %>%
  gather("Algorithm", "Variance", 2:4)

df2_r <- df2 %>% select(time, ab_r, ucb_r, mix_r) %>%
  rename(`A/B Testing`=ab_r, `UCB`=ucb_r, `Hybrid_alg`=mix_r) %>%
  gather("Algorithm", "Squared Error of Variance", 2:4)



df2_s <- df2 %>% select(time, ab_s, ucb_s, mix_s) %>% gather("type", "sum", 2:4)

ggplot(df2_m, aes(time, mean)) + geom_line(aes(colour=type))

ggplot(df2_v, aes(time, variance)) + geom_line(aes(colour=type))




df3 <- read.csv("analysis_trt.csv")  

df3_m <- df3 %>% select(time, ab_m, ucb_m, mix_m) %>% gather("type", "mean", 2:4) 
df3_v <- df3 %>% select(time, ab_v, ucb_v, mix_v) %>%
  rename(`A/B Testing`=ab_v, `UCB`=ucb_v, `Hybrid_alg`=mix_v) %>%
  gather("Algorithm", "Variance", 2:4)

df3_r <- df3 %>% select(time, ab_r, ucb_r, mix_r) %>%
  rename(`A/B Testing`=ab_r, `UCB`=ucb_r, `Hybrid_alg`=mix_r) %>%
  gather("Algorithm", "Squared Error of Variance", 2:4)

ggplot(df3_m, aes(time, mean)) + geom_line(aes(colour=type))

ggplot(df3_v, aes(time, variance)) + geom_line(aes(colour=type))

ggplot() + geom_line(data = df2_v, aes(time, Variance, colour=Algorithm)) +
  geom_line(data = df3_v, aes(time, Variance, colour=Algorithm)) +
  theme_bw() + ggsave('out_var.png', width = 8, height = 5)

ggplot() + geom_line(data = df2_r, aes(time, `Squared Error of Variance`, colour=Algorithm)) +
  geom_line(data = df3_r, aes(time, `Squared Error of Variance`, colour=Algorithm)) +
  theme_bw() + ggsave('out_err.png', width = 8, height = 5)

