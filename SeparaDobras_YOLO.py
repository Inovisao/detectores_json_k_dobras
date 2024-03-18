import json
import os
import shutil
import yaml

def CriarLabelsYOLOV8(fold):

    with open('dataset/all/train/_annotations.coco.json', 'r') as f:
        anotacaoGeral = json.load(f)
    
    diretorio = "dataset/YOLO"

    #Verifica se o diretorio existe e romove ele
    if os.path.exists(diretorio):  
        shutil.rmtree(diretorio)         
    classes = anotacaoGeral['categories']
    className = []
    #Ira pegar as classes do dataset
    for i in range(len(classes)):
        className.append(classes[i]['name'])
    # Defina o caminho para o arquivo YAML que você deseja alterar
    caminho_arquivo_yaml = 'data.yaml'

    # Carregue o conteúdo do arquivo YAML
    with open(caminho_arquivo_yaml, 'r') as arquivo:
        conteudo = yaml.safe_load(arquivo)

    # Altere o conteúdo conforme necessário
    conteudo['nc'] = len(className)
    conteudo['names'] = className

    # Salve o conteúdo alterado de volta no arquivo YAML
    with open(caminho_arquivo_yaml, 'w') as arquivo:
        yaml.dump(conteudo, arquivo)

    caminho_train_labels = 'dataset/YOLO/train/labels'
    caminho_test_labels = 'dataset/YOLO/test/labels'
    caminho_valid_labels = 'dataset/YOLO/valid/labels'
    caminho_train_images = 'dataset/YOLO/train/images'
    caminho_test_images = 'dataset/YOLO/test/images'
    caminho_valid_images = 'dataset/YOLO/valid/images'
    os.makedirs(caminho_train_labels,exist_ok=True)
    os.makedirs(caminho_test_labels,exist_ok=True)
    os.makedirs(caminho_valid_labels,exist_ok=True)
    os.makedirs(caminho_test_images,exist_ok=True)
    os.makedirs(caminho_train_images,exist_ok=True)
    os.makedirs(caminho_valid_images,exist_ok=True)

    caminhos = (os.listdir('dataset/filesJSON'))
    foldsUsadas = []
    #Pega o caminho do arquivo coco que esta sendo usada
    for caminho in caminhos:
        if str(caminho[0:6]) == str(fold):
            foldsUsadas.append(caminho)

    for Caminho in foldsUsadas:
        path = Caminho[7:-5]

        caminho = 'dataset/filesJSON/'+Caminho
        # Lendo o arquivo JSON
        with open(caminho, 'r') as f:
            anotacaoDobras = json.load(f)
        # Exibindo os dados lidos
        NameFile = []
        imageID = []
        anotacao = {}
        idAnotcao = {}
        #Salva a lista de id das imagens
        for i in range(len(anotacaoDobras['annotations'])):
            imageID.append(anotacaoDobras['annotations'][i]['image_id'])

        #Faz a lista sem repetir os IDs
        imageID = (list(set(imageID)))

        #Cria um dicionario Com ID das imagens
        for i in imageID:
            anotacao[i] = []
            idAnotcao[i] = [] 
        #Ira pegar as anotações es classes de cada imagens e salvar em seus dicionarios
        for id in imageID:
            for i in range(len(anotacaoDobras['annotations'])):
                if anotacaoDobras['annotations'][i]['image_id'] == id:
                    anotacao[id].append(anotacaoDobras['annotations'][i]['bbox'])
                    idAnotcao[id].append(anotacaoDobras['annotations'][i]['category_id']-1)
        #Salva o nome de cada imagem
        for i in imageID:
            for j in range(len(anotacaoDobras['images'])):
                if (anotacaoDobras['images'][j]['id']) == i:
                    NameFile.append(anotacaoDobras['images'][j]['file_name'])
        
        slectImage = 0
        #Converte as anotações para o formato da YOLOV8
        for id in imageID:
            linhas = []
            for i in range (len(anotacao[id])):
                x1 = int(anotacao[id][i][0])
                y1 = int(anotacao[id][i][1])

                x2 = int(x1 + int(anotacao[id][i][2]))
                y2 = int(y1 + int(anotacao[id][i][3]))

                x_center = abs((x1+ x2)/2)
                y_center = abs((y1 + y2) / 2)

                width = abs(anotacao[id][i][2])
                height = abs(anotacao[id][i][3])
                linhas.append(str(idAnotcao[id][i])+' '+str(x_center/640)+' '+str(y_center/640)+' '+str(width/640)+' '+str(height/640)+"\n")

            image = 'dataset/all/train/'+NameFile[slectImage]
            arq = NameFile[slectImage][0:-4]+'.txt'
            with open(arq, 'w') as arquivo:
            # Escrevendo múltiplas linhas no arquivo
                arquivo.writelines(linhas)
            slectImage+=1
            if path == 'test':
                shutil.move(arq, caminho_test_labels)
                shutil.copy(image,caminho_test_images)
            elif path == 'val':
                shutil.move(arq, caminho_valid_labels)
                shutil.copy(image,caminho_valid_images)
            else:
                shutil.move(arq, caminho_train_labels)
                shutil.copy(image,caminho_train_images)