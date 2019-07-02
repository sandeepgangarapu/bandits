library(ggplot2)
library(dplyr)
library(tidyr)
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\MAB_VAR")


df <- read.csv("compare_output.csv")

sum(df$b_arm==df$ab_barm)
sum(df$b_arm==df$mix_barm)
sum(df$b_arm==df$ucb_barm)
sum(df$b_arm==df$eps_barm)
df$mix_var_barm[df$mix_barm!=df$ucb_barm]
