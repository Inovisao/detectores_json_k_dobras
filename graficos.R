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
library("Metrics")

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BOXPLOT DO DESEMPENHO ENTRE TÉCNICAS
#
dados <- read.table('./dataset/results.csv',sep=',',header=TRUE)
metricas <- list("mAP50","MAE","RMSE","r")
graficos <- list()
i <- 1

for (metrica in metricas) {
   print(metrica)
   TITULO = sprintf("Boxplot for %s",metrica)

      g <- ggplot(dados, aes_string(x=dados$ml, y=metrica,fill=dados$ml)) + 
           geom_boxplot(alpha=0.3)+
#           scale_fill_brewer(palette="Purples")+
           labs(title=TITULO,x="ML Technique", y = metrica)+
           theme(legend.position="none",plot.title = element_text(hjust = 0.5))
   
   graficos[[i]] <- g
   i = i + 1
   print(g)
}

g <- grid.arrange(grobs=graficos, ncol = 2)
ggsave(paste("./dataset/boxplot.png", sep=""),g, width = 10, height = 10)
print(g)



# -------------------------------------------------------------------
# -------------------------------------------------------------------
# CURVAS DE APRENDIZAGEM
#

nets <- levels(dados$ml) 
contaDobras <- dados[dados$ml == nets[1], ]

DOBRAS=nrow(contaDobras)

# GAMBIARRA PARA PEGAR O TOTAL DE ÉPOCAS DE DENTRO DE experimento.py 
log <- readLines('experimento.py')
log <-log[grepl('EPOCAS=',log)]
logTable <- read.table(text = log,sep='=')
EPOCAS=logTable[1,2]

# GAMBIARRA PARA ENCONTRAR E PEGAR DADOS DO ARQUIVO DE LOG
# SE TIVER MAIS DE UM ARQUIVO .log DÁ ERROR POR ISSO TEM
# QUE LIMPAR OS RESULTADOS ANTERIORES ANTES DE RODAR O
# EXPERIMENTO
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



# -------------------------------------------------------------------
# -------------------------------------------------------------------
# XY CONTAGEM MANUAL X AUTOMÁTICA
#
dados <- read.table('./dataset/counting.csv',sep=',',header=TRUE)

graficos <- list()
i <- 1


for (net in nets) {

   filtrado <- dados[dados$ml == net, ]

   RMSE = rmse(filtrado$groundtruth,filtrado$predicted)
   MAE = mae(filtrado$groundtruth,filtrado$predicted)
   R = cor(filtrado$groundtruth,filtrado$predicted)
   TITULO = sprintf("%s RMSE = %.3f MAE =  %.3f R = %.3f",net,RMSE,MAE,R)

   g <- ggplot(filtrado, aes(x=groundtruth, y=predicted)) + 
        geom_point()+
        geom_smooth(method='lm')+
        labs(title=TITULO,x="Measured", y = "Predicted")

   
   graficos[[i]] <- g
   i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 1)
ggsave(paste("./dataset/counting.png", sep=""),g, width = 10, height = 8)
print(g)


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# HISTOGRAMA DA DISTRIBUIÇÃO DOS DADOS DO CONJUNTO DE TESTE
# (CONTAGENS MANUAIS)

ggplot(filtrado, aes(x=groundtruth))+
   geom_histogram(color="darkblue", fill="lightblue")+
   xlab("Objects Countings")+
   ylab("Density")+
   ggtitle("Histogram & Density Curve for Ground Truth Countings")



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



