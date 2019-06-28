library(ggplot2)
library(dplyr)
library(tidyr)
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\MAB_VAR")


df <- read.csv("output.csv")

df$x = 1:nrow(df)
df_regret <- df %>% select(x, ab_regret, ucb_regret, mix_regret) %>%
  gather("type", "regret", 2:4)

ggplot(df_regret) + geom_point(aes(x=x, y=regret, color=type))+ ggsave("regret.png.png")

df_mean_rmse <- df %>% select(x, ab_mean_rmse, ucb_mean_rmse, mix_mean_rmse) %>%
  gather("type", "m_rmse", 2:4)

ggplot(df_mean_rmse) + geom_point(aes(x=x, y=m_rmse, color=type)) + ggsave("mean_rmse.png.png")

df_var_rmse <- df %>% select(x, ab_var_rmse, ucb_var_rmse, mix_var_rmse) %>%
  gather("type", "var_rmse", 2:4)

ggplot(df_var_rmse) + geom_point(aes(x=x, y=var_rmse, color=type)) + ggsave("var_rmse.png.png")


df <- read.csv("groups.csv")
df$x = 1:nrow(df)
df_group <- df %>% select(x, ab_group, ucb_group, mix_group) %>% gather("type", "group", 2:4)
ggplot(df_group) + geom_point(aes(x=x, y=factor(group))) + facet_grid(type ~.) + ggsave("group.png")



