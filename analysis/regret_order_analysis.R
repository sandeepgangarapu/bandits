suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(data.table)
  })
require(gridExtra)

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


data <- fread("wise_regret_analysis.csv") 

data <- data[, alg := ifelse(alg=="ab", "ABTesting", ifelse(alg=="thomp","THOMP", ifelse(alg=="thomp_inf", "THOMP_INF", NA)))]
group_outcome <- data[, x:= seq_len(.N),.(ite,alg)][, .(alg, regret, x)]
group_outcome_thomp <- group_outcome[alg=='THOMP']
group_outcome_thompinf <- group_outcome[alg=='THOMP_INF']

group_filter <- group_outcome[x==20 | x==200 | x==2000 | x==20000 | x==200000]


logest <- lm(log(regret)~log(x),data=group_outcome_thompinf)
summary(logest)

plot(log(group_outcome_thompinf$x), log(group_outcome_thompinf$regret))
plot(log(group_outcome_thomp$x), group_outcome_thomp$regret)
