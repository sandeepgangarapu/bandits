suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(dplyr)
    library(tidyr)
  })
require(gridExtra)


setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


data <- read.csv("wise_mse_graph.csv") %>% mutate(alg = ifelse(alg=="ab", "ABTesting", ifelse(alg=="thomp","THOMP", ifelse(alg=="thomp_inf", "THOMP_INF", NA) )))
#data <- read.csv("bias_1_2000.csv")



group_outcome <- data %>% 
  select(alg, group, ite, outcome) %>%
  group_by(ite,alg) %>% mutate(x=row_number())  %>% 
  ungroup()

regret_mse <- data %>% 
  select(alg, regret, ite, mean_mse, var_mse) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>% 
  ungroup()



xlimit <- max(group_outcome$x) + 500


# Setting new WD so that results are saved in different folder
setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\results")



df <- group_outcome %>% filter(ite==0) 



grp = ggplot(df) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(alg ~.) +
  labs(x='Time Period', y = 'Arm') +
  theme(axis.text.y = element_text(size = rel(0.7)))  +  theme_bw() + 
  #ggsave("grp.png", dpi=400, height = 4, width=8, scale = 0.8)


df_regret = regret_mse %>% filter(ite==0)


reg <- ggplot(df_regret, aes(x=x, y=regret)) + geom_line(aes(color=alg)) +
  xlim(0,2300) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) + theme_minimal() + 
  theme(legend.position = "none") + 
  labs(x='Time Period', y = 'Regret') 
  #ggsave("wise_reg.png", dpi=400, height = 4, width=8, scale = 0.8)




ggsave("wise_grp_reg.png", arrangeGrob(grp, reg, nrow = 1, ncol=2),
       dpi=400, height = 4, width=12, scale = 0.9)

df_mse <- regret_mse %>% select(-c(regret, var_mse)) %>% rename(mse = mean_mse) %>% filter(ite==0)



# %>% 
#   group_by(x, alg) %>%
#   summarise(mn_mse = mean(mse), se_mse = sd(mse)/sqrt(n()))


ggplot(df_mse, aes(x=x, y=mse)) + geom_line(aes(color=alg)) +
  xlim(0,2150) + ylim(0,0.6) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) + theme_minimal() + 
  theme(legend.position = "none") + 
  labs(x='Time Period', y = 'MSE of Mean') + 
  ggsave("wise_mn.png", dpi=400, height = 5, width=10, scale = 0.8)


df_mse <- regret_mse %>% select(-c(regret, var_mse)) %>% rename(mse = mean_mse) %>% filter(ite==0)

# %>% 
#   group_by(x, alg) %>%
#   summarise(mn_mse = mean(mse), se_mse = sd(mse)/sqrt(n()))


mn <- ggplot(df_mse, aes(x=x, y=mse)) + geom_line(aes(color=alg)) +
  xlim(0,2500) + ylim(0,1.15) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) + theme_minimal() + 
  theme(legend.position = "none") + 
  labs(x='Time Period', y = 'MSE of Mean') 


df_mse <- regret_mse %>% select(-c(regret, mean_mse)) %>% rename(mse = var_mse) %>% filter(ite==0)



var <- ggplot(df_mse, aes(x=x, y=mse)) + geom_line(aes(color=alg)) +
  xlim(0,2500) + ylim(0,1.8) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) + theme_minimal() + 
  theme(legend.position = "none") + 
  labs(x='Time Period', y = 'MSE of Variance') 

par(mfrow = c(1,2))

ggsave("wise_mn_var.png", arrangeGrob(mn, var, nrow = 1, ncol=2),
       dpi=400, height = 4, width=10, scale = 0.8)

