setwd("G:\\My Drive\\Research\\Contextual Bandits\\code\\bandits\\r_simulations")
source("distributions.R")

# caluclate power

mean(y_0)
mean(y_1)
mean(y_2)

library(pwr)
pwr.t.test(d=1, type = "two.sample", alternative = "greater", power = 0.8)
# just need 13 samples for y_1
pwr.t.test(d=3, type = "two.sample", alternative = "greater", power = 0.8)
# just need 3 samples for y_2


# lets conduct experiment using this power calculation

