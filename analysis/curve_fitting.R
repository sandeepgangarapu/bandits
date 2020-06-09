setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")

df <- read.csv("sim_1_10000.csv")

regret_mse <- df %>% select(alg, regret, group) %>%
  group_by(alg) %>% mutate(x=row_number()) %>%
  ungroup()

log_10_func <- function(x){10*log(x+1)}
log_200_func <- function(x){350*log((x/50)+1)}
sqrt_func = function(x){17*sqrt((1/10) *x *log(x+1))}

ggplot(regret_mse, aes(x=x, y=regret)) + geom_line(aes(color=alg)) +
 # stat_function(fun = log_10_func) +
  #stat_function(fun = log_200_func) +
  stat_function(fun = sqrt_func) +
  labs(title = "Regret") + xlim(0,11100) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) +
  theme_bw() 



ggplot(regret_mse) + geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(alg ~.) +
  theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = rel(0.7))) 

