suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(dplyr)
    library(directlabels)
    library(gganimate)
    library(cumstats)
  })

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")

true_means = c(0.25, 1.82, 1.48, 2.25, 2)
true_vars = c(2.84,  1.97, 2.62, 1, 2.06)

algs = c('ab', 'eps_greedy', 'ucb', 'ucb_inf_eps',
          'thomp', 'thomp_inf_eps')


df_main <- read.csv("sim_hypo_3.csv")
num_ite <- max(df_main$ite)
#algs <- unique(df_main$alg)

res <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(res) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
for (a in algs){
  for (i in c(1:num_ite)){
  print(a)
  print(i)
  df = df_main %>%filter(alg==a & ite==i)
  ttest = pairwise.t.test(df$outcome, df$group,
                      p.adjust.method = "BH")
  pval = ttest$p.value
  print(pval)
  de<-data.frame(a, i, pval[1], pval[2], pval[4])
  colnames(de) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
  res <- rbind(res, de)
  }
}


res <- res %>% mutate(pval_1 = ifelse(pval_1<0.05, 1, 0),
                      pval_2 = ifelse(pval_2<0.05, 1, 0),
                      pval_3 = ifelse(pval_3<0.05, 1, 0))

res1 <- res %>% group_by(alg) %>% summarise(pval_1_eff = sum(pval_1)/n(), 
                                            pval_2_eff = sum(pval_2)/n(),
                                            pval_3_eff = sum(pval_3)/n())



df_main <- read.csv("sim_hypo_same_mean_same_var.csv")
num_ite <- max(df_main$ite)
#algs <- unique(df_main$alg)

res <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(res) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
for (a in algs){
  for (i in c(1:num_ite)){
    print(a)
    print(i)
    df = df_main %>%filter(alg==a & ite==i)
    ttest = pairwise.t.test(df$outcome, df$group,
                            p.adjust.method = "BH")
    pval = ttest$p.value
    print(pval)
    de<-data.frame(a, i, pval[1], pval[2], pval[4])
    colnames(de) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
    res <- rbind(res, de)
  }
}


res <- res %>% mutate(pval_1 = ifelse(pval_1<0.05, 1, 0),
                      pval_2 = ifelse(pval_2<0.05, 1, 0),
                      pval_3 = ifelse(pval_3<0.05, 1, 0))

res2 <- res %>% group_by(alg) %>% summarise(pval_1_eff = sum(pval_1)/n(), 
                                            pval_2_eff = sum(pval_2)/n(),
                                            pval_3_eff = sum(pval_3)/n())



df_main <- read.csv("sim_hypo_same_mean_diff_var.csv")
num_ite <- max(df_main$ite)
#algs <- unique(df_main$alg)

res <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(res) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
for (a in algs){
  for (i in c(1:num_ite)){
    print(a)
    print(i)
    df = df_main %>%filter(alg==a & ite==i)
    ttest = pairwise.t.test(df$outcome, df$group,
                            p.adjust.method = "BH")
    pval = ttest$p.value
    print(pval)
    de<-data.frame(a, i, pval[1], pval[2], pval[4])
    colnames(de) <- c('alg', 'ite', 'pval_1', 'pval_2', 'pval_3')
    res <- rbind(res, de)
  }
}


res <- res %>% mutate(pval_1 = ifelse(pval_1<0.05, 1, 0),
                      pval_2 = ifelse(pval_2<0.05, 1, 0),
                      pval_3 = ifelse(pval_3<0.05, 1, 0))

res3 <- res %>% group_by(alg) %>% summarise(pval_1_eff = sum(pval_1)/n(), 
                                            pval_2_eff = sum(pval_2)/n(),
                                            pval_3_eff = sum(pval_3)/n())

