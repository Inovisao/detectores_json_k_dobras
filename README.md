# Detectores_json_k_dobras
__Autores__: Código cedido pelo Prof. Jonathan Andrade Silva (UFMS) 

Adaptações feitas por Hemerson Pistori (pistori@ucdb.br) e Marcelo Kuchar (marcelokuchar@gmail.com)

__Versao__: 1.0.1 

__Objetivo__: Facilitar a aplicação de validação cruzada em k dobras e gerar resultados da aplicação 
de diversos detectores do pacote mmdetection em uma banco de imagens anotadas pelo Roboflow no formato COCO Json


### Instalação e dependências:

#### Para rodar no Google Colab 

- Entre no site https://colab.research.google.com/
- Crie uma conta no Colab caso ainda não tenha (dá para usar a do seu gmail)
- Utilize a opção Upload e suba o arquivo experimento_colab.ipynb
- Siga as instruções que estão dentro deste arquivo e que vão aparecer no Colab


#### Para rodar na sua própria máquina:

Testado no Ubuntu 20.04 com python 3.7

Leia o arquivo **install.sh** para ver o que é preciso instalar
(a criação do ambiente conda deve ser feito FORA do script de instalação) 

mmdetection 2.12.0 - incluso no git 
(não funciona para versões mais recentes do mmdetection)

Placa gráfica: GTX 3060
Driver nvidia: 460



### Preparação dos dados
 
- Use o Roboflow para anotar e coloque tudo como treinamento e gerar um arquivo compactado (.zip) com as imagens e anotações. Não divida em treinamento, validação e teste dentro do roboflow pois isso será feito fora do roboflow, em um esquema de validação cruzada em dobras. No canal do youtube do Prof. Pistori tem um vídeo sobre o Roboflow

```
google-chrome https://roboflow.com/
```

- Baixe o arquivo gerado no roboflow para a pasta do detectores_json_k_dobras
- Apague a pasta ./dataset/all/train (se ela existir)
- Mova para a pasta ./dataset/all o arquivo do roboflow e o descompacte lá
- Será criada uma pasta train contendo as imagens e as anotações 
- Comandos que podem ser utilizados para realizar estas operações:

```
mv suas_imagens_roboflow.zip ./dataset/all
cd ./dataset/all/  
rm -rf train
unzip suas_imagens_roboflow.zip
```

- Rode o utilitário que cria as dobras de treino, teste e validação. Altere o número de dobras (-folds) e o percentual de validação (-valperc) se necessário. Os resultados desta etapa ficarão na pasta ./dataset/filesJSON 

```
cd ../../utils
conda activate detectores # Leia antes o arquivo install.sh para criar o ambiente conda
./apagaResultados.sh  
python geraDobras.py -folds=5 -valperc=0.3  # Gera as dobras para a validação cruzada 
```

- Troque dentro do arquivo experimento.py, nas linhas 10 e 11, o nome da classe que você usou para anotar suas imagens (no lugar de **eucaliptos**) e o número de dobras, caso não tenha utilizado 5

- Na pasta ./utils existem alguns outros script que podem ser úteis em algumas situações. É preciso estudá-los antes de usá-los para bagunçar seus dados.



### Escolhendo as arquiteturas a serem testadas
# 

Procure no arquivo experimento.py o lugar onde criamos a variável MODELS_CONFIG e leia com atenção as orientações. É preciso baixar os arquivos .pth das redes que serão utilizadas e colocar dentro da pasta ./checkpoints. Estes arquivos estão disponíveis no site do mmdetection e no repositório do Inovisão (acesso restrito). 

Os arquivos .pth para rede vfnet, por exemplo, podem ser encontrados no link:
https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/README.md . Dentro este site procure por um link chamado 'model' (podem ter vários, para as várias versões da rede que você pode escolher).

Você encontrará as outras redes aqui (tem que baixar todas que for usar):
https://github.com/open-mmlab/mmdetection/tree/master/configs


### Rodando o experimento 

```
$ . ./conda_init.sh
$ python experimento.py
```

### Gerando os gráficos

- Pode ser necessário instalar alguns pacotes. Neste caso, veja o arquivo install_R_packages.R

- Se o total de épocas foi alterado em **experimento.py** é preciso alterar TAMBÉM em **graficos.R**

```
$ Rscript graficos.R
```


### Encontrando os resultados
# 

Gráficos e arquivos .csv:
- Na pasta dataset são criados 2 gráficos:
  - boxplot.png : boxplot comparando o desempenho das técnicas)
  - history.png : curvas de aprendizagem usando conjunto de validação)
- Estes gráficos são gerados a partir destes dois arquivos:
  - results.csv : resultados por técnica, tamanho da caixa e dobra
  - epocas.csv : evolução da perda no conjunto de validação durante o treinamento

Arquivo de LOG:
- Um único arquivo de log com informações sobre todas as redes e o
  histórico da aprendizagem é salvo na primeira dobra na pasta
  correspondente ao primeiro modelo treinado. Exemplo:
  ./dataset/fold_1/MModels/vfnet_r50/20210423_180309.log
- Também são salvos arquivos de log no formato .json  separadamente
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

