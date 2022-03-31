# __compara_detectores
__autores__: Código cedido pelo Prof. Jonathan Andrade Silva (UFMS) 
             1 - Adaptações feitas por Hemerson Pistori (pistori@ucdb.br)
             2 - Adaptações feitas por Marcelo Kuchar (marcelokuchar@gmail.com)
__versao__: 1.0.1 

Objetivo: Facilitar a aplicação de validação cruzada em k dobras e gerar resultados da aplicação 
de diversos detectores do pacote mmdetection em uma banco de imagens anotadas pelo Roboflow no formato COCO Json


### Instalação e dependências:

#### Para rodar no Google Colab 

- Entre https://colab.research.google.com/
- Crie uma no conta no Colab caso ainda não tenha (dá para usar a do seu gmail)
- Utilize a opção Upload e suba o arquivo MMdetection_Colab_Drive.ipynb
- Siga as instruções que estão dentro deste arquivo e que vão aparecer no Colab


#### Para rodar na sua própria máquina:

Testado no Ubuntu 20.04 com python 3.7

Leia o arquivo install.sh para ver o que é preciso instalar
(a criação do ambiente conda deve ser feito FORA do script de instalação) 

mmdetection 2.12.0 - incluso no git 
(não funciona para versões mais recentes do mmdetection)

Placa gráfica: GTX 3060
Driver nvidia: 460

Install.sh #instala as dependências.



### Preparação dos dados
# 
- Use o Roboflow para anotar e coloque tudo como treinamento (não divida em validação e teste)
- Gere o .zip e coloque na pasta ./dataset/all 
- Descompacte dentro da pasta all (vai criar uma subpasta chamada train). Se já existe um pasta train, apague o que tem dentro (na versão atual tem um exemplo pronto com imagens de Eucaliptos lá dentro)
- Rode o utilitário que vai preparar os dados e separar nas dobras 

$ cd dataset/all/  
$ unzip exemplo_anotacoes_roboflow.zip
$ cd ../../utils
$ ./preparaDadosPelaPrimeiraVez.sh # Não esqueça antes de criar e iniciar seu ambiente CONDA (ler install.sh). Veja primeiro o que tem dentro deste arquivo pois pode não ser necessário rodá-lo.

- Se necessário, troque o número 4 pela quantidade de dobras que você quiser
  e o 0.3 pelo percentual que você quiser usar para o conjunto de validação
  (este percentual é em relação ao que sobra depois que tirar o conjunto de teste)
  DENTRO DO ARQUIVO preparaDadosPelaPrimeiraVez.sh 
  procure por "python geraDobras.py -folds=4 -valperc=0.3"

- Os arquivos das anotações gerados ficarão na pasta ./dataset/filesJSON
  (Confira para ver se gerou mesmo)

O script preparadaDadosPelaPrimeiraVez.sh chama outros dois script para
fazer um ajuste relativo a base que usamos para testar (ovos de aedes)
- Trocar o nome categoria de ovo para Corn # Daria também para trocar
                                             dentro código Corn por ovo
- Remove a categoria zero (precisa ter uma única categoria)

CUIDADO: Estes ajustes podem não ser necessários para o seu banco de imagens. 
Neste caso, você pode rodar estes dois comandos aqui separadamente

$ ./apagaResultados.sh  # Apaga arquivos gerados na última execução do treinamento
$ python geraDobras.py -folds=4 -valperc=0.3  # Gera as dobras para a validação cruzada 


### Escolhendo as arquiteturas a serem testadas
# 
Procure no arquivo experimento.py o lugar onde criamos a variável 
MODELS_CONFIG e leia com atenção as orientações. Ajuste também o total
de épocas e dobras neste arquivo, se necessário. Se alterar o total
de épocas, arrume também a variável EPOCAS dentro de graficos.R

É preciso também baixar o arquivo .pth no site do mmdetection e colocar dentro da
pasta ./checkpoints. Os arquivos .pth para rede vfnet, por exemplo, podem ser
encontrados no link abaixo:
https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/README.md
Dentro do site acima procure por um link chamado 'model' (podem ter vários, para as várias versões da rede que você pode escolher)

Você encontrará as outras redes aqui (tem que baixar todas que for usar):
https://github.com/open-mmlab/mmdetection/tree/master/configs

O Inovisão tem um link para vários destes arquivos que são mais usados pelo grupo (consultem o grupo pelo whatsapp)


### Rodando o treinamento e os testes
# 

$ . ./conda_init.sh
$ ./roda.sh

-- Dentro do arquivo roda.sh tem uma chamada para o programa em R (graficos.R) que está comentado, você pode descomentar ou então rodar este comando individualmente depois de treinar as redes

### Encontrando os resultados
# 

Gráficos e arquivos .csv:
- Na pasta dataset são criados 2 gráficos:
  - boxplot.png : boxplot comparando o desempenho das técnicas)
  - history.png : curvas de aprendizagem usando conjunto de validação)
- Este gráficos são gerados a partir destes dois arquivos:
  - results.csv : resultados por técnica, tamanho da caixa e dobra
  - epocas.csv : evolução da perda no conjunto de validação durante o treinamento

Arquivo de LOG:
- Um único arquivo de log com informações sobre todas as redes e o
  histórico da aprendizagem é salvo na primeira dobra na pasta
  correspondente ao primeiro modelo treinado. Exemplo:
  ./dataset/fold_1/MModels/vfnet_r50/20210423_180309.log
- Também são salvos artigos de log no formato .json  separadamente
  para cada dobra e rede utilizada. Exemplo:
  dataset/fold_3/MModels/atss_r50/20210424_061001.log.json
 
  
Imagens com Resultados:
- Na pasta ./dataset são criadas subpastas com o prefixo
  "prediction_" contendo cada imagem do conjunto de teste com o
  resultado das detecções mostranda com retângulos em verde

Precisão e Recall são calculados sobre as predições com confiança >= 50% e com 0.3 IOU sobre uma caixa verdade (caixa anotada manualmente) para modificar esses valores encontre as linhas 449 e 455 em experimentos.py


Outras informações:
- Os pesos da rede, os hyperparâmetros usados, etc também são gravados nas pastas 
  ./dataset/fold_X/MModels

