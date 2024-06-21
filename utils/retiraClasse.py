# retiraClasse 
# Autor: Hemerson Pistori 
# Descrição: troca no arquivo json em ../dataset/all/train/_annotations.coco.json 
# os nomes das classes por um mesmo nome, passado como parâmetro.
# O arquivo original é mantido e um novo arquivo json é criado na pasta ../utils/

from pycocotools.coco import COCO
import argparse
import json

parser = argparse.ArgumentParser(description='Troca nomes das classes')

parser.add_argument('-annotations', default='../dataset/all/train/_annotations.coco.json',  type=str,  help='Caminho para o arquivo com as anotações',required=False)
parser.add_argument('-classe', default='target',type=str, help='Nome da nova classe única',required=False)

args = parser.parse_args()

anotacoes = COCO(args.annotations)

# Mantem apenas a categoria zero, trocanco o nome para o passado como parâmetro
categorias = anotacoes.dataset['categories']
categorias[0]['name'] = args.classe
categorias[0]['id'] = 1

# Remove as outras categorias
categorias = categorias[0:1]

# Troca a categoria para 0 em todas as anotações
for anotacao in anotacoes.dataset['annotations']:
    anotacao['category_id'] = 1

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

# Salva o arquivo com as novas anotações
save_coco('../utils/_annotations.coco.json', anotacoes.dataset['info'], anotacoes.dataset['licenses'], 
          anotacoes.dataset['images'], anotacoes.dataset['annotations'], categorias)

print('\n\n')
print('Arquivo salvo em ../utils/_annotations.coco.json')
print('Classes alteradas para: ',args.classe)
print('NÃO ESQUEÇA DE COPIAR O ARQUIVO PARA ../dataset/all/train/_annotations.coco.json')