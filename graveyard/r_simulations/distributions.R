# setting seed for replicability
set.seed(123)


num_arms <- 3             #NUmber of treatments = 2
n_rows <- 1000            # number of subjects

# covariates

male <- sample(0:1, n_rows, replace = TRUE) 
age <- rnorm(n_rows, 30, 20)

# control group outome is set to normal for now. So we can just play with treatment effects
y_0 = rnorm(n_rows, 0, 1)

tau_1=c()

for (i in male) {
  if (i==1) {
    tau_1 <- c(tau_1, rnorm(1, 3, 1))
  }  else {
    tau_1 <- c(tau_1, rnorm(1, -1, 1))
  }
}

tau_2=c()

for (i in male) {
  if (i==1) {
    tau_2 <- c(tau_2, rnorm(1, 1, 3))
  }  else {
    tau_2 <- c(tau_2, rnorm(1, 5, 2))
  }
}

y_2 = y_0 + tau_2
y_1 = y_0 + tau_1

# Checking distributions
hist(y_0)
hist(y_1)
hist(y_2)

# Checking normality of 
# null hypothesis is normal
shapiro.test(y_0)
shapiro.test(y_1) #not normal
shapiro.test(y_2) #not_normal


# mann-whitney test to check between hetrogeneity
# null hypothesis is that both are same
wilcox.test(y_1, y_0) # not same
wilcox.test(y_2, y_0) # not same
wilcox.test(y_2, y_1)  # not same

# flinger test to check variance difference
fligner.test(y_1, y_0)
fligner.test(y_1, y_2)
fligner.test(y_2, y_0)


