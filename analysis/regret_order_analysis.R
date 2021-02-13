suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(data.table)
  })
require(gridExtra)

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


data <- fread("wise_regret_analysis_100.csv") 

data <- data[, alg := ifelse(alg=="ab", "ABTesting", ifelse(alg=="thomp","THOMP", ifelse(alg=="thomp_inf", "THOMP_INF", NA)))]
group_outcome <- data[, x:= seq_len(.N),.(ite,alg)][, .(alg, regret, x)][,regret:=mean(regret),.(alg, x)]
group_outcome_thomp <- group_outcome[alg=='THOMP']
group_outcome_thompinf <- group_outcome[alg=='THOMP_INF']


logest <- lm(log(regret)~log(x),data=group_outcome_thompinf)
summary(logest)



logest <- lm(log(regret)~log(x),data=group_outcome_thomp)
summary(logest)


plot(log(group_outcome_thompinf$x), log(group_outcome_thompinf$regret))
plot(log(group_outcome_thompinf$x), (group_outcome_thompinf$regret))

plot(log(group_outcome_thomp$x), group_outcome_thomp$regret)

plot(log(group_outcome_thomp$x), log(group_outcome_thomp$regret))

