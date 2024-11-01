# Roda experimentos com as mesmas redes neurais, mas com diferentes hiperparâmetros
#
# Os resultados vão ficar em uma nova pasta chamada resultados

lrS=(0.001)
lcS=(0.1 0.01)
liS=(0.1 0.01)

rm -rf resultados
mkdir -p resultados

for lr in ${lrS[@]}; do
    for lc in ${lcS[@]}; do
        for li in ${liS[@]}; do
            echo 'Rodando para lr=' $lr ' lc=' $lc ' li=' $li
            python experimento.py -lr=$lr -lc=$lc -li=$li
            Rscript graficos.R
            mkdir -p resultados/lr$lr-lc$lc-li$li
            cp -R dataset ./resultados/lr$lr-lc$lc-li$li
            cp nohup.out ./resultados/lr$lr-lc$lc-li$li/
            cp experimento.py ./resultados/lr$lr-lc$lc-li$li/
        done
    done
done


