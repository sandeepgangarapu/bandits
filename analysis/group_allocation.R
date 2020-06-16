suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(directlabels)
    library(gganimate)
    library(cumstats)
  })

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")

true_means = c(1.5,2.8,3)

true_vars = c(1,1,1)

data <- read.csv("group_alloc_1_500_2.csv")

group_outcome <- data %>% 
  select(alg, group, ite, outcome) %>%
  group_by(ite,alg) %>% mutate(x=row_number()) %>%
  ungroup()


plot_ab <- ggplot(group_outcome %>% filter(alg %in% c('ab'))) +
  geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(alg ~.) +
  theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = rel(0.7))) 


plot_thomp <- ggplot(group_outcome %>% filter(alg %in% c('thomp'))) +
  geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(alg ~.) +
  theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = rel(0.7))) 

plot_thomp_inf <- ggplot(group_outcome %>% filter(alg %in% c('thomp_inf_eps'))) +
  geom_point(aes(x=x, y=factor(group)), shape=1, alpha=0.6) +
  facet_grid(alg ~.) +
  theme_bw() +
  labs(x='Time Period', y = 'Group') +
  theme(axis.text.y = element_text(size = rel(0.7))) 
# plot_thomp <- plot_thomp + transition_time(x) + shadow_mark()
# animate(plot_thomp, fps = 10, width = 900, height = 200)
# anim_save("alloc_thomp.gif")
# plot_ab <- plot_ab + transition_time(x) + shadow_mark()
# animate(plot_ab, fps = 10, width = 900, height = 200)
# anim_save("alloc_ab.gif")
plot_thomp_inf <- plot_thomp_inf + transition_time(x) + shadow_mark()
animate(plot_thomp_inf, fps = 10, width = 900, height = 200)
anim_save("alloc_thomp_inf.gif")