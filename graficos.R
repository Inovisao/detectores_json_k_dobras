# Se precisar carregar pacotes adicionais, siga os exemplos abaixo 
#install.packages("psych")

library("ggplot2")
library("gridExtra")
library("plyr")
library("stringr")
library("forcats")
library("scales")
library("forcats")
library("ExpDes")
library("dplyr")
library("ExpDes.pt")
library(tidyr)

EPOCAS=20


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BOXPLOT DO DESEMPENHO ENTRE TÉCNICAS
#
dados <- read.table('./dataset/results.csv',sep=',',header=TRUE)
metricas <- list("AP50")
graficos <- list()
i <- 1

for (metrica in metricas) {

   print(metrica)
   TITULO = sprintf("Boxplot for %s",metrica)

      g <- ggplot(dados, aes(x=ml, y=AP50,fill=ml)) + 
           geom_boxplot(alpha=0.3)+
           #scale_fill_brewer(palette="Purples")+
           labs(title=TITULO,x="ML Technique", y = metrica)+
           theme(legend.position="none",plot.title = element_text(hjust = 0.5))
   
   graficos[[i]] <- g
   i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 1)
ggsave(paste("./dataset/boxplot.png", sep=""),g, width = 10, height = 8)
print(g)



# -------------------------------------------------------------------
# -------------------------------------------------------------------
# CURVAS DE APRENDIZAGEM
#

nets <- levels(dados$ml) 
contaDobras <- dados[dados$ml == nets[1], ]

DOBRAS=nrow(contaDobras)


logFile <- list.files(".", "log$", recursive=TRUE, full.names=TRUE, include.dirs=TRUE)
log <- readLines(logFile)
epocas <-log[grepl('- mmdet - INFO - Epoch\\(',log)]
epocas <- gsub("[,:\\[]", " ", epocas)
epocas <- gsub("[]]", " ", epocas)
epocas <- gsub("loss ", ",", epocas)

epocasVal <- read.table(text = epocas,sep=',')

#epocasVal <- epocasVal[ , c("V2")]
colnames(epocasVal) <- c("rest","loss")

folds <- sprintf("fold_%d",seq(1:DOBRAS))
epochs <- 1:EPOCAS

novasColunas <- tidyr::crossing(nets,folds,epochs)

dados <- cbind(novasColunas,epocasVal)
write.csv(dados,'./dataset/epocas.csv')

# Pegando apenas dados da primeira dobra 
filtrado <- dados[dados$folds == "fold_1", ]
TITULO = sprintf("Validation loss evolution during training")
g <- ggplot(filtrado, aes(x=epochs, y=loss, colour=nets, group=nets)) +
    geom_line() +
    ggtitle(TITULO)+
    theme(plot.title = element_text(hjust = 0.5))


ggsave(paste("./dataset/history.png", sep=""),g)
print(g)




#dados <- read.table('../results_dl/resultados.csv',sep=',',header=TRUE)
#
#sink('../results_dl/two_way.txt')
#
#cat(sprintf('\n\n====>>> TESTANDO: PRECISÃO =============== \n\n',metrica))
#fat2.dic(dados$architecture, dados$optimizer, dados$precision, quali = c(TRUE,TRUE), mcomp="sk") 
#cat(sprintf('\n\n====>>> TESTANDO: RECALL ================= \n\n',metrica))
#fat2.dic(dados$architecture, dados$optimizer, dados$recall, quali = c(TRUE,TRUE), mcomp="sk") 
#cat(sprintf('\n\n====>>> TESTANDO: FSCORE ================= \n\n',metrica))
#fat2.dic(dados$architecture, dados$optimizer, dados$fscore, quali = c(TRUE,TRUE), mcomp="sk") 

#sink()



