suppressPackageStartupMessages(
  {
    library(ggplot2)
    library(data.table)
  })
require(gridExtra)
require(directlabels)

setwd("G:\\My Drive\\Research\\Bandits\\code\\bandits\\analysis\\output")


df1 <- fread("xi_analysis0.6.csv") 
df2 <- fread("xi_analysis0.8.csv") 
df3 <- fread("xi_analysis1.csv") 
df4 <- fread("xi_analysis1.2.csv") 
df5 <- fread("xi_analysis1.4.csv") 

df1[, `:=`(xi=0.6, x= seq_len(.N))]
df2[, `:=`(xi=0.8, x= seq_len(.N))]
df3[, `:=`(xi=1, x= seq_len(.N))]
df4[, `:=`(xi=1.2, x= seq_len(.N))]
df5[, `:=`(xi=1.4, x= seq_len(.N))]


main_df <- rbind(df1, df2, df3, df4, df5)

ggplot(main_df, aes(x=x, y=regret)) + geom_line(aes(color=factor(xi))) +
  xlim(0,2400) +
  geom_dl(aes(label=alg), method=list('last.points', cex=0.8)) + theme_minimal() + 
  theme(legend.position = "none") + 
  labs(x='Time Period', y = 'Regret') +
  ggsave("xi_analysis.png", dpi=400, height = 4, width=8, scale = 0.8)

