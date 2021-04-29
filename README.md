# __compara_detectores
__autores__: Código cedido pelo Prof. Jonathan Andrade Silva (UFMS)
             Adaptações feitas por Hemerson Pistori (pistori@ucdb.br)
__versao__: 1.0.0 

Objetivo: Facilitar a aplicação de validação cruzada em k dobras e gerar resultados da aplicação 
de diversos detectores do pacote mmdetection em uma banco de imagens anotadas pelo Robodlow no formato COCO Json

### Instalação e dependências:

Testado no Ubuntu 20.04 com python 3.7.10
Leia o arquivo install.sh para ver o que é preciso instalar
(a criação do ambiente conda deve ser feito FORA do script de instalação) 

Placa gráfica: GTX 1070
Driver nvidia: 450.66


### Preparação dos dados
# 
- Use o Roboflow para anotar e coloque tudo como treinamento (não divida em validação e teste)
- Gere o .zip e coloque na pasta ./dataset/all (TEM UM ARQUIVO DE EXEMPLO LÁ)
- Descompacte dentro da pasta all (vai criar uma subpasta chamada train)
- Rode o utilitário que vai preparar os dados e separar nas dobras

$ cd dataset/all/
$ unzip exemplo_anotacoes_roboflow.zip
$ 

$ cd utils
$ ./preparaDadosPelaPrimeiraVez.sh

- Se necessário, troque o número 4 pela quantidade de dobras que você quiser
  e o 0.3 pelo percentual que você quiser usar para o conjunto de validação
  (este percentual é em relação ao que sobra depois que tirar o conjunto de teste)
  DENTRO DO ARQUIVO preparaDadosPelaPrimeiraVez.sh 
  procure por "python geraDobras.py -folds=4 -valperc=0.3"

- Os arquivos das anotações gerados ficarão na pasta ./dataset/filesJSON
  (Confira para ver se geral mesmo)

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


### Rodando o treinamento e os testes
# 

$ . ./conda_init.sh
$ ./roda.sh

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

Outras informações:
- Os pesos da rede, os hyperparâmetros usados, etc também são gravados nas pastas 
  ./dataset/fold_X/MModels

