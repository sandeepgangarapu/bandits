library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)
library(ggpubr)
#install.packages("directlabels", repo="http://r-forge.r-project.org")
library(lattice)
library(directlabels)


setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\MAB_VAR")


df <- read.csv("output.csv")

df$x = 1:nrow(df)
df_regret <- df %>% select(x, ab_regret, ucb_regret, eps_regret, ucb_inf_var_prop_regret)  %>%
  rename(`AB-TESTING` = ab_regret,
         `UCB` = ucb_regret,
         `E-GREEDY` = eps_regret,
         `UCB-INF` = ucb_inf_var_prop_regret) %>%
  gather("type", "regret", 2:5)

ggplot(df_regret %>%  filter(type!="E-GREEDY")) +
  geom_line(aes(x=x, y=regret, color=type))+ 
  theme_minimal()+ 
  theme(legend.position = "none") +
  labs(x='No. of Allocations', y = 'Regret', color="Allocation Type")+
  ggsave("regret.png", width = 10, height = 4,scale=0.7, units="in")

df_mean_mse <- df %>% select(x, ab_mean_mse, ucb_mean_mse, eps_mean_mse, ucb_inf_var_prop_mean_mse) %>%
  rename(`AB-TESTING` = ab_mean_mse,
         `UCB` = ucb_mean_mse,
         `E-GREEDY` = eps_mean_mse,
         `UCB-INF` = ucb_inf_var_prop_mean_mse) %>%
  gather("type", "mean_mse", 2:5)

ggplot(df_mean_mse %>%  filter(type!="E-GREEDY"), aes(x=x, y=mean_mse, color=type)) + geom_point() + 
  theme_minimal()+ 
  theme(legend.position = "none") +
  labs(x='No. of Allocations', y = 'MSE of Mean Estimate', color="Allocation Type") +  
  ggsave("mean_mse.png", width = 6, height = 5, scale = 0.7, units="in")


df_var_mse <- df %>% select(x, ab_var_mse, ucb_var_mse, eps_var_mse, ucb_inf_var_prop_var_mse) %>%
  rename(`AB-TESTING` = ab_var_mse,
         `UCB` = ucb_var_mse,
         `E-GREEDY` = eps_var_mse,
         `UCB-INF` = ucb_inf_var_prop_var_mse) %>%
  gather("type", "var_mse", 2:5)

ggplot(df_var_mse %>%  filter(type!="E-GREEDY")) + geom_point(aes(x=x, y=var_mse, color=type))+ 
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x='No. of Allocations', y = 'MSE of Variance Estimate', color="Allocation Type") +
  ggsave("var_mse.png", width = 6, height = 5, scale = 0.7, units="in")


df <- read.csv("groups.csv")
df$x = 1:nrow(df)
df_group <- df %>% select(x, ab_group, ucb_group, eps_group, ucb_inf_var_prop_group) %>% rename(`AB-TESTING` = ab_group,
                                                                                                `UCB` = ucb_group,
                                                                                                `E-GREEDY` = eps_group,
                                                                                                `UCB-INF` = ucb_inf_var_prop_group) %>% 
  gather("type", "group", 2:5)

ggplot(df_group) + geom_line(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(type ~.) + theme_bw() +
  labs(x='Time Period', y = 'Allocated Group') +
  theme(axis.text.y = element_text(size = 6)) +
  ggsave("group.png", width = 8, height = 4)

ggplot(df_group %>% filter(type %in% c('AB-TESTING'))) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(type ~.) + theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = 6)) +   theme(axis.text.y = element_text(size = rel(1)))  +
  ggsave("group_ab.png", width = 10, height = 2, dpi=300, units="in")

ggplot(df_group %>% filter(type %in% c('UCB'))) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(type ~.) + theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = 6)) +   theme(axis.text.y = element_text(size = rel(1)))  +
  ggsave("group_ucb_2.png", width = 8, height = 2, dpi=300, units="in")


ggplot(df_group %>% filter(type %in% c('UCB-INF'))) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) + facet_grid(type ~.) + theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = 6)) +   theme(axis.text.y = element_text(size = rel(1)))  +
  ggsave("group_ucbinf.png", width = 8, height = 2, dpi=300, units="in")



# 
# ggplot(df_group %>% filter(type=='UCB')) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6)  + theme_bw() +
#   labs(x='Time Period', y = 'Allocated Group') +
#   theme(axis.text.y = element_text(size = 6)) +
#   ggsave("group_ucb.png", width = 8, height =2)
# 
# ggplot(df_group %>% filter(type=='UCB-VAR')) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6)  + theme_bw() +
#   labs(x='Time Period', y = 'Allocated Group') +
#   theme(axis.text.y = element_text(size = 6)) +
#   ggsave("group_ucbvar.png", width = 8, height =2)
# 
# 
# ggplot(df_group %>% filter(type=='AB_testing')) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6)  + theme_bw() +
#   labs(x='Time Period', y = 'Allocated Group') +
#   theme(axis.text.y = element_text(size = 6)) +
#   ggsave("group_ab.png", width = 8, height =2)



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