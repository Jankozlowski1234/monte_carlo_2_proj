setwd("C:/Users/jaako/Desktop/studia/Monte Carlo/monte_carlo_2_proj")


library(ggplot2)
library(dplyr)

#black shoes formula
r <- 0.05
sigma<-0.25
S<-100
K<-100

d1<-1/sigma*(log(S/K)+r+sigma^2/2)
d2<-d1-sigma

BSF<-S*pnorm(d1)-K*exp(-r)*pnorm(d2)






#first plot 

df1 <- read.csv("data_n_1.csv",header = T)

df1$n<-as.factor(df1$n)
df1$R<-as.factor(df1$R)

#df1<-filter(df1, abs(value-12.5) <2.5)

ggplot(df1, aes(x=type, y=value )) + 
  geom_boxplot()


wyk1<-df1 %>%
  group_by(type)%>%
  summarise(variance = var(value),
            means =  mean(value),
            low = means-qnorm(1-0.05/2)*sqrt(variance/length(value)),
            up = means+qnorm(1-0.05/2)*sqrt(variance/length(value)))

wyk1<-data.frame(wyk1)


ggplot(wyk1, aes(x=type, y = means)) + geom_point() +  
  geom_errorbar(aes(ymin = low, ymax = up)) + 
  geom_hline(yintercept=BSF, linetype="dashed", color = "red")+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "95% przedziały ufności różnych estymatorów dla n=1",
       subtitle = "z dodaną prawdzową wartością",
       x="Estymatory",y=" ",
       caption = "Wykres 1")



# second plot 

df22 <- read.csv("data_n_1_different_R2.csv",header = T)

wyk22<-df22 %>%
  group_by(type,R)%>%
  summarise(variance = var(value),
            means =  mean(value))

ggplot(wyk22,aes(x=R,y=variance,col = type))+geom_line()+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "Wariancje różnych estymatorów dla n=1",
       subtitle = "dla różnych R",
       x="R",y="Wariancja",
       caption = "Wykres 2")

#third plot 
df3 <- read.csv("data_n_different.csv",header = T)

wyk3<-df3 %>%
  group_by(type,n)%>%
  summarise(variance = var(value),
            means =  mean(value),
            low = means-qnorm(1-0.05/2)*sqrt(variance/length(value)),
            up = means+qnorm(1-0.05/2)*sqrt(variance/length(value)))

wyk3<-data.frame(wyk3)

ggplot(wyk3, aes(x=n, y = means,col = type)) + geom_point(alpha = 0.7) +  
  geom_errorbar(aes(ymin = low, ymax = up))+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "95% przedziały ufności różnych estymatorów",
       subtitle = "dla różnych n",
       x="n",y="",
       caption = "Wykres 2")

ggplot(wyk3,aes(x=n,y=variance,col = type))+geom_line(alpha = 0.7)+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "Wariancje różnych estymatorów",
       subtitle = "dla różnych n",
       x="R",y="Wariancja",
       caption = "Wykres 2")


ggplot(wyk3,aes(x=n,y=means,col = type))+geom_line(alpha = 0.7)+
  theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))+ 
  labs(title = "Średnie różnych estymatorów",
       subtitle = "dla różnych n",
       x="R",y="Średnia",
       caption = "Wykres 2")





