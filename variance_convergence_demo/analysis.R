#install.packages("cumstats")
setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\variance_convergence_demo")

library(cumstats)
library(ggplot2)

#Hypothesis
# Three scenarios
# 1. Same mean, different variance
# 2. Same variance, different mean (small variance)
# 3. Same variance, different mean (large variance)

# For scenario 1, we expect the error of variance estimate i.e (sigma - sugma_hat)
# to converge faster if variance value is less

# For scenarios 2,3 it should not matter


x = c(1:100)
a = rnorm(100,0,1)
b = rnorm(100,0,4)
ca = cumvar(a)
cb = cumvar(b)
e_ca = 1-ca
e_cb = 16-cb
df = data.frame(x=x, e_ca = e_ca, e_cb=e_cb)
ggplot(df) + geom_line(aes(x,e_ca), color='blue') + geom_line(aes(x,e_cb), color='red') +
  ggtitle("Same mean, different variance") + ggsave("Same mean different variance.png")


x = c(1:100)
a = rnorm(100,1,1)
b = rnorm(100,16,1)
ca = cummean(a)
cb = cummean(b)
e_ca = 1-ca
e_cb = 16-cb
df = data.frame(x=x, e_ca = e_ca, e_cb=e_cb)
ggplot(df) + geom_line(aes(x,e_ca), color='blue') + geom_line(aes(x,e_cb), color='red')+
  ggtitle("Same variance, different mean (small variance)") + ggsave("Same variance different mean (small variance).png")

x = c(1:100)
a = rnorm(100,1,4)
b = rnorm(100,16,4)
ca = cummean(a)
cb = cummean(b)
e_ca = 1-ca
e_cb = 16-cb
df = data.frame(x=x, e_ca = e_ca, e_cb=e_cb)
ggplot(df) + geom_line(aes(x,e_ca), color='blue') + geom_line(aes(x,e_cb), color='red')+
  ggtitle("Same variance, different mean (large variance)") + ggsave("Same variance different mean (large variance).png")


# all hypothesis are proven
# In Graph 1 red and blue lines are clearly not following similar path even on different seeds.
# Graph 2 and 3 are following similar path for different seeds 
