library(dplyr)
library(ggplot2)
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\peeking")
df <- read.csv("ucb_peek.csv")

ggplot(df) + geom_point(aes(x=time, y=factor(ucb_p_group))) 
ggplot(df) + geom_point(aes(x=time, y=factor(ucb_v_group)), colour = "red")
