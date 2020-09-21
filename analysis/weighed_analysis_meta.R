def_func <- function(df, true_df){
  
  est_algs <- c("thomp_ipw", "thomp_aipw", "thomp_eval_aipw",
                "thomp_inf_ipw", "thomp_inf_aipw", "thomp_inf_eval_aipw")
  thomp_algs <- c( "thomp", "thomp_inf")
  all_thomp <- c(est_algs, thomp_algs)
  means <- df %>% mutate(mn = mean_est)
  true_means <- true_df$true_means
  
  
  # bias of the worst arm
  worst_bias <- means %>% filter(alg %in% all_thomp) %>% filter(group==which.min(true_means)-1) %>% 
    mutate(bias = mn-min(true_means)) %>% group_by(alg) %>%
    summarise(Bias = mean(bias), se = sd(bias)/sqrt(n())) %>% mutate(low=Bias-qnorm(0.975)*se,
                                                                     high=Bias+qnorm(0.975)*se)
  


  #variance of the worst arm
  worst_var <- means %>% filter(alg %in% all_thomp) %>% filter(group==which.min(true_means)-1) %>% 
    group_by(alg) %>%  summarise(var = var(mn))
  
  
  # things to check
  
  # for default algs
  thomp_neg_bias <- subset(worst_bias, alg=='thomp', select="Bias") < 0
  thomp_inf_neg_bias <- subset(worst_bias, alg=='thomp', select="Bias") < 0
  # beacuse bias is negative, the sign should change in the below expression
  thomp_inf_less_thomp_bias <- subset(worst_bias, alg=='thomp_inf', select="Bias") > subset(worst_bias, alg=='thomp', select="Bias")
  # ipw removes bias
  thomp_bias_remo_ipw_low <- subset(worst_bias, alg=='thomp_ipw', select="low") < 0
  thomp_bias_remo_ipw_high <- subset(worst_bias, alg=='thomp_ipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_bias_remo_ipw <- !xor(thomp_bias_remo_ipw_low, thomp_bias_remo_ipw_high)
  
  # aipw removes bias
  thomp_bias_remo_aipw_low <- subset(worst_bias, alg=='thomp_aipw', select="low") < 0
  thomp_bias_remo_aipw_high <- subset(worst_bias, alg=='thomp_aipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_bias_remo_aipw <- !xor(thomp_bias_remo_aipw_low, thomp_bias_remo_aipw_high)
  
  
  # athey removes bias
  thomp_bias_remo_eval_aipw_low <- subset(worst_bias, alg=='thomp_eval_aipw', select="low") < 0
  thomp_bias_remo_eval_aipw_high <- subset(worst_bias, alg=='thomp_eval_aipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_bias_remo_eval_aipw <- !xor(thomp_bias_remo_eval_aipw_low, thomp_bias_remo_eval_aipw_high)
  
  
  # Samething for thompson inf
  # ipw removes bias
  thomp_inf_bias_remo_ipw_low <- subset(worst_bias, alg=='thomp_inf_ipw', select="low") < 0
  thomp_inf_bias_remo_ipw_high <- subset(worst_bias, alg=='thomp_inf_ipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_inf_bias_remo_ipw <- !xor(thomp_inf_bias_remo_ipw_low, thomp_inf_bias_remo_ipw_high)
  
  # aipw removes bias
  thomp_inf_bias_remo_aipw_low <- subset(worst_bias, alg=='thomp_inf_aipw', select="low") < 0
  thomp_inf_bias_remo_aipw_high <- subset(worst_bias, alg=='thomp_inf_aipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_inf_bias_remo_aipw <- !xor(thomp_inf_bias_remo_aipw_low, thomp_inf_bias_remo_aipw_high)
  
  
  # athey removes bias
  thomp_inf_bias_remo_eval_aipw_low <- subset(worst_bias, alg=='thomp_inf_eval_aipw', select="low") < 0
  thomp_inf_bias_remo_eval_aipw_high <- subset(worst_bias, alg=='thomp_inf_eval_aipw', select="high") > 0
  # if bias is removed then confidence intervals will cover 0 and the signs of the above two vars will be different
  thomp_inf_bias_remo_eval_aipw <- !xor(thomp_inf_bias_remo_eval_aipw_low, thomp_inf_bias_remo_eval_aipw_high)
  
  
  
  # variance is less for thompson inf compared to thomp
 
  thomp_inf_less_thomp_var <- subset(worst_var, alg=='thomp_inf', select="var") < subset(worst_var, alg=='thomp', select="var")
  
  
  # variance is high for ipw compared to thomp
  thomp_ipw_high_thomp_var <- subset(worst_var, alg=='thomp_ipw', select="var") > subset(worst_var, alg=='thomp', select="var")
  
  
  # variance of aipw is less compared to ipw 
  thomp_aipw_less_thomp_ipw_var <- subset(worst_var, alg=='thomp_aipw', select="var") < subset(worst_var, alg=='thomp_ipw', select="var")
  
  
  # variance of eval aipw is less compated to ipw and aipw

  thomp_eval_aipw_less_thomp_aipw_var <- subset(worst_var, alg=='thomp_eval_aipw', select="var") < subset(worst_var, alg=='thomp_aipw', select="var")
  thomp_eval_aipw_less_thomp_ipw_var <- subset(worst_var, alg=='thomp_eval_aipw', select="var") < subset(worst_var, alg=='thomp_ipw', select="var")
  
  
  
  # variance is high for ipw compared to thomp
  thomp_ipw_high_thomp_var <- subset(worst_var, alg=='thomp_ipw', select="var") > subset(worst_var, alg=='thomp', select="var")
  
  
  # variance of aipw is less compared to ipw 
  thomp_aipw_less_thomp_ipw_var <- subset(worst_var, alg=='thomp_aipw', select="var") < subset(worst_var, alg=='thomp_ipw', select="var")
  
  
  # variance of eval aipw is less compated to ipw and aipw
  
  thomp_eval_aipw_less_thomp_aipw_var <- subset(worst_var, alg=='thomp_eval_aipw', select="var") < subset(worst_var, alg=='thomp_aipw', select="var")
  thomp_eval_aipw_less_thomp_ipw_var <- subset(worst_var, alg=='thomp_eval_aipw', select="var") < subset(worst_var, alg=='thomp_ipw', select="var")
  
  
  # variance is high for ipw compared to thomp
  thomp_inf_ipw_high_thomp_inf_var <- subset(worst_var, alg=='thomp_inf_ipw', select="var") > subset(worst_var, alg=='thomp', select="var")
  
  
  # variance of aipw is less compared to ipw 
  thomp_inf_aipw_less_thomp_inf_ipw_var <- subset(worst_var, alg=='thomp_inf_aipw', select="var") < subset(worst_var, alg=='thomp_inf_ipw', select="var")
  
  
  # variance of eval aipw is less compated to ipw and aipw
  
  thomp_inf_eval_aipw_less_thomp_inf_aipw_var <- subset(worst_var, alg=='thomp_inf_eval_aipw', select="var") < subset(worst_var, alg=='thomp_inf_aipw', select="var")
  thomp_inf_eval_aipw_less_thomp_inf_ipw_var <- subset(worst_var, alg=='thomp_inf_eval_aipw', select="var") < subset(worst_var, alg=='thomp_inf_ipw', select="var")
  
  
  out <- list('thomp_neg_bias' = thomp_neg_bias[1,],
              'thomp_inf_neg_bias' = thomp_inf_neg_bias[1,],
              'thomp_inf_less_thomp_bias' = thomp_inf_less_thomp_bias[1,],
              'thomp_bias_remo_ipw' = thomp_bias_remo_ipw[1,],
              'thomp_bias_remo_aipw' = thomp_bias_remo_aipw[1,],
              'thomp_bias_remo_eval_aipw' = thomp_bias_remo_eval_aipw[1,],
              'thomp_inf_bias_remo_ipw' = thomp_inf_bias_remo_ipw[1,],
              'thomp_inf_bias_remo_aipw' = thomp_inf_bias_remo_aipw[1,],
              'thomp_inf_bias_remo_eval_aipw' = thomp_inf_bias_remo_eval_aipw[1,],
              'thomp_inf_less_thomp_var' = thomp_inf_less_thomp_var[1,],
              'thomp_ipw_high_thomp_var' = thomp_ipw_high_thomp_var[1,],
              'thomp_aipw_less_thomp_ipw_var' = thomp_aipw_less_thomp_ipw_var[1,],
              'thomp_eval_aipw_less_thomp_aipw_var' = thomp_eval_aipw_less_thomp_aipw_var[1,],
              'thomp_eval_aipw_less_thomp_ipw_var' = thomp_eval_aipw_less_thomp_ipw_var[1,],
              'thomp_inf_ipw_high_thomp_inf_var' = thomp_inf_ipw_high_thomp_inf_var[1,],
              'thomp_inf_aipw_less_thomp_inf_ipw_var' = thomp_inf_aipw_less_thomp_inf_ipw_var[1,],
              'thomp_inf_eval_aipw_less_thomp_inf_aipw_var' = thomp_inf_eval_aipw_less_thomp_inf_aipw_var[1,],
              'thomp_inf_eval_aipw_less_thomp_inf_ipw_var' = thomp_inf_eval_aipw_less_thomp_inf_ipw_var[1,])
  out <- data.frame(out)
  return(out)
            
}


func_normality <- function(df){
  means <- df %>% mutate(mn = mean_est) %>% group_by(alg) %>%
    summarise(test = shapiro.test(mn)$p.value) %>%
    mutate(test_pass = ifelse(test > 0.05, "Yes", "No"))
  return(means)
}



setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output\\weighed_meta")

file_name <- "sim_weighed_ite_1000_t_500_metaite_"

grp = c(0:(length(true_means)-1))
true_df <- data.frame(grp, true_means, true_vars)
true_df <- true_df %>% mutate(grp=factor(grp))

ref1 <- read.csv("ref.csv") %>% drop_na()
ref2 <- read.csv("ref_1.csv") %>% drop_na()
ref <- rbind(ref1, ref2) %>% rename(true_means = mn, true_vars = vr)
grp <- rep(c(0:5), 100)
grp <- grp[1:nrow(ref)]
ref$grp <- factor(grp)

df <- data.frame()
norm <- data.frame()
for (i in c(c(0:27), c(60:87))) {
  print(i)
  fname <- paste(file_name, i, ".csv", sep='')
  data <- read.csv(fname) %>% drop_na() %>%
    filter(mean_est != -Inf) %>% filter(mean_est != Inf)
  ref_df <- ref %>% filter(ite==i)
  sub_df <- def_func(data, ref_df)
  df <- rbind(df, sub_df)
  df_final <- df %>% pivot_longer(1:18, names_to = "alg") %>% group_by(alg, value) %>% summarise(cnt = n())
  normality <- func_normality(data)
  norm <- rbind(norm, normality)
}
