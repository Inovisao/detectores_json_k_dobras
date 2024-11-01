# Busca todos os arquivos results.csv da pasta resultados (e subpastas) e junta em um único arquivo
# chamado allResults.csv
# Retira o cabeçalho de todos os arquivos results.csv e junta em um único arquivo chamado allResultsSemCabecalho.csv

echo "hyper,ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore" > temp.csv

# Para cada arquivo results.csv encontrado, retira o cabeçalho e junta no arquivo allResults.csv
# trocar o valor da primeira coluna pelo nome da pasta onde o arquivo foi encontrado
listaArquivos=$(find resultados -name results.csv)

for arquivo in $listaArquivos; do
    # Pega a string entre resultados/ e /dataset
    pasta=$(echo $arquivo | sed -n 's/.*resultados\/\(.*\)\/dataset.*/\1/p')
    tail -n +2 $arquivo | sed "s/^/$pasta,/" >> temp.csv
    echo 'Processando ' $arquivo
done

# Junta colunas 1 e 2
awk -F, '{print $2"_"$1","$3","$4","$5","$6","$7","$8","$9","$10","$11","$12}' temp.csv > allResults.csv

# Troca ml_hyper por ml no cabeçalho de allResults.csv
sed -i 's/ml_hyper/ml/' allResults.csv


# Faz a mesma coisa mas para os arquivos counting.csv
echo "hyper,ml,fold,groundtruth,predicted,TP,FP,dif,fileName" > temp.csv

# Para cada arquivo counting.csv encontrado, retira o cabeçalho e junta no arquivo allCounting.csv
# trocar o valor da primeira coluna pelo nome da pasta onde o arquivo foi encontrado
listaArquivos=$(find resultados -name counting.csv)

for arquivo in $listaArquivos; do
    echo 'Processando ' $arquivo
    # Pega a string entre resultados/ e /dataset
    pasta=$(echo $arquivo | sed -n 's/.*resultados\/\(.*\)\/dataset.*/\1/p')
    tail -n +2 $arquivo | sed "s/^/$pasta,/" >> temp.csv
done

# Junta colunas 1 e 2
awk -F, '{print $2"_"$1","$3","$4","$5","$6","$7","$8","$9}' temp.csv > allCounting.csv

# Troca ml_hyper por ml no cabeçalho de allCounting.csv
sed -i 's/ml_hyper/ml/' allCounting.csv

mv allResults.csv ./dataset/results.csv
mv allCounting.csv ./dataset/counting.csv

rm -rf dataset/fold_*
rm -rf dataset/prediction*

find resultados -name all | xargs rm -rf
find resultados -name filesJSON | xargs rm -rf 

Rscript graficos.R


