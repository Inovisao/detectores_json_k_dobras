# Troca no arquivo de anotações as palavras Ovos e Ovo por Corn (o código está usando Corn)

file=../dataset/all/train/_annotations.coco.json
sed -i 's/ovos/Corn/g' $file 
sed -i 's/ovo/Corn/g' $file



