library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)

setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\MAB_VAR")


df <- read.csv("output.csv")

df$x = 1:nrow(df)
df_regret <- df %>% select(x, ab_regret, ucb_regret, mix_prop_regret, trt_prop_mix_regret)  %>%
  rename(`AB_testing` = ab_regret,
         `UCB` = ucb_regret,
         `UCB-VAR` = mix_prop_regret,
         `UCB_TRT_VAR` = trt_prop_mix_regret) %>%
  gather("type", "regret", 2:5)

ggplot(df_regret) + geom_line(aes(x=x, y=regret, color=type))+ theme_bw()+
  labs(x='Time Period', y = 'Regret', color="Allocation Type")+ ggsave("regret.png", width = 8, height = 4)

df_mean_rmse <- df %>% select(x, ab_mean_rmse, ucb_mean_rmse, mix_prop_mean_rmse, trt_prop_mix_mean_rmse) %>%
  rename(`AB_testing` = ab_mean_rmse,
         `UCB` = ucb_mean_rmse,
         `UCB-VAR` = mix_prop_mean_rmse,
         `UCB_TRT_VAR` = trt_prop_mix_mean_rmse) %>%
  gather("type", "mean_rmse", 2:5)

ggplot(df_mean_rmse) + geom_point(aes(x=x, y=mean_rmse, color=type), size=0.5)+ labs(x='Time Period', y = 'RMSE of Mean')+ theme_bw() +
  ggsave("mean_rmse.png", width = 6, height = 4)

df_var_rmse <- df %>% select(x, ab_var_rmse, ucb_var_rmse, mix_prop_var_rmse, trt_prop_mix_var_rmse) %>%
  rename(`AB_testing` = ab_var_rmse,
         `UCB` = ucb_var_rmse,
         `UCB-VAR` = mix_prop_var_rmse,
         `UCB_TRT_VAR` = trt_prop_mix_var_rmse) %>%
  gather("type", "var_rmse", 2:5)

ggplot(df_var_rmse) + geom_point(aes(x=x, y=var_rmse, color=type), size=0.5)+ theme_bw()+ guides(colour = guide_legend(override.aes = list(size = 2))) +
  labs(x='Time Period', y = 'RMSE of Variance', color="Allocation Type")+ ggsave("var_rmse.png", width = 8, height = 4)


df <- read.csv("groups.csv")
df$x = 1:nrow(df)
df_group <- df %>% select(x, ab_group, ucb_group, mix_prop_group, trt_mix_prop_group) %>% rename(`AB_testing` = ab_group,
                                                                                   `UCB` = ucb_group,
                                                                                   `UCB-VAR` = mix_prop_group,
                                                                                   `UCB_TRT_VAR` = trt_mix_prop_group) %>% 
  gather("type", "group", 2:5)

ggplot(df_group) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(type ~.) + theme_bw() +
  labs(x='Time Period', y = 'Allocated Group') +
  theme(axis.text.y = element_text(size = 6)) +
  ggsave("group.png", width = 8, height = 4)

# 
# df <- read.csv("var.csv")
# df_var <- df %>% select(arm, ab_var, ucb_var, eps_var, mix_var) %>% rename(`AB_testing` = ab_var,
#                                                                                     `UCB` = ucb_var,
#                                                                                    `e-greedy` = eps_var,
#                                                                                    `UCB-VAR` = mix_var) %>% 
#   gather("type", "var", 2:5) 
# df_var[is.na(df_var)] <- 0
# 
# df_var$x = rep(rep(1:2000, each=10),4)
# true_var = c(4.03757288,3.43165603,0.46557525,0.9803696,3.53352318,3.54460421, 4.56793121,1.04190155, 0.7491534,  3.0408862)
# df_var$true_var = rep(true_var, 8000)
# 
# df_var$arm = paste("arm ",df_var$arm)
# ggplot(df_var) + geom_point(aes(x=x, y=var, color=type), size=0.3) + geom_hline(aes(yintercept=true_var)) +
#   facet_wrap(~arm, nrow=5) + theme_bw() + 
#   labs(x='Time Period', y = 'Variance Estimate', color="Allocation Type") +  theme(legend.position="bottom")+
#   theme(axis.text.y = element_text(size = 6)) + guides(colour = guide_legend(override.aes = list(size = 2))) +
#   ggsave("var.png", width = 8, height = 10) 
# 
