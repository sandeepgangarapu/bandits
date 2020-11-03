setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


true_means = c(0, 0.2)


#data <- read.csv("hyp_ite_310_t_1052.csv") %>% drop_na() 
#data <- read.csv("hyp_ite_310_t_170.csv") %>% drop_na() 
data <- read.csv("hyp_ite_1000_t_2000.csv") %>% drop_na() 

data_0 <- data %>% filter(group==3) 
data_1 <- data %>% filter(group==4)

a <- data_0[data_0$alg=='thomp_eval_aipw_var', 'var_est']
data_0[data_0$alg=='thomp_eval_aipw', 'var_est'] = a

a <- data_0[data_0$alg=='thomp_inf_eval_aipw_var', 'var_est']
data_0[data_0$alg=='thomp_inf_eval_aipw', 'var_est'] = a

a <- data_1[data_1$alg=='thomp_eval_aipw_var', 'var_est']
data_1[data_1$alg=='thomp_eval_aipw', 'var_est'] = a

a <- data_1[data_1$alg=='thomp_inf_eval_aipw_var', 'var_est']
data_1[data_1$alg=='thomp_inf_eval_aipw', 'var_est'] = a


merged_data <- merge(data_0, data_1, by=c('ite','alg'))

hyp_data <- merged_data %>% 
  mutate(tval = (mean_est.y-mean_est.x)/(sqrt((var_est.x/arm_pull_tracker.x)+(var_est.y/arm_pull_tracker.y))),
         df = arm_pull_tracker.x+arm_pull_tracker.y-2,
         pval = dt(tval, df), test_pass = ifelse(pval<0.05, TRUE, FALSE))

final_results <- hyp_data %>% filter(!grepl("var", alg) & test_pass !=TRUE) %>%
  group_by(alg, test_pass) %>% summarise(cnt = n(), perc = n()/1000)

