setwd("C:/Users/jaako/Desktop/studia/Monte Carlo/monte_carlo_2_proj")

library(ggplot2)


df <- read.csv("rys1.csv",header = T)

df$strata<-as.factor(df$strata)

ggplot(df,aes(x = x,y = y,color = strata))+geom_point()


df1 <- read.csv("data_n_1.csv",header = T)

df1$n<-as.factor(df1$n)
df1$R<-as.factor(df1$R)

ggplot(df1, aes(x=type, y=value )) + 
  geom_boxplot()

ggplot(subset(df1,df1$type %in% c("CMC","Control","Antithetic")), aes(x=type, y=value )) + 
  geom_boxplot()

df1[df1$value<10]




