# Código que irá treinar as redes, executar os testes e gravar os resultados
# Autor: Cedido pelo Prof. Jonathan Andrade Silva (UFMS)
#        Pequenas adaptações feitas por Hemerson Pistori (pistori@ucdb.br)


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# DEFINE ALGUNS HIPERPARÂMETROS
#

DOBRAS=4
EPOCAS=100
TAXA_APRENDIZAGEM=0.1

# Coloque True se quiser apenas testar redes previamente treinada
APENAS_TESTA=False

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# MOSTRA A VERSÃO DAS PRINCIPAIS BIBLIOTECAS
import torch, torchvision
print('Torch: ',torch.__version__, torch.cuda.is_available())
import mmdet
print('mmdet: ',mmdet.__version__)
import mmcv
print('mmvc: ', mmcv.__version__)
#import pycocotools
#print('Pycoco:',pycocotools.__version__)
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print('cuda-version: ', get_compiling_cuda_version())
print('compiler: ', get_compiler_version())
import sys
print('python: ',sys.version)



from mmcv import Config
import os
import sys
import time
import matplotlib
import matplotlib.pylab as plt
from mmdet.datasets import build_dataset,build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet import __version__
from mmdet.utils import collect_env, get_root_logger
from mmcv.utils import get_git_hash
import torch
import os.path as osp
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets import CocoDataset
from mmcv.visualization import color_val
from mmdet.core import visualization as vis
import mmcv, torch
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math





sys.path.append('mmdetection')

pasta_mmdetection=os.path.join(os.getcwd(),'mmdetection')
pasta_dataset=os.path.join(os.getcwd(),'dataset')
pasta_checkpoints=os.path.join(os.getcwd(),'checkpoints')
print('Pasta do mmdetection: ',pasta_mmdetection)
print('Pasta com o dataset: ',pasta_dataset)
print('Pasta com os checkpoints (*.pth): ',pasta_checkpoints)

plt.rcParams["axes.grid"] = False



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# DEFINE AS REDES QUE SERÃO USADAS
#
# Se for usar alguma rede diferente das que já estão em MODELS_CONFIG 
# baixo é preciso retirar o comentário ou acrescentar
# mais linhas copiando das que já existem e alterando o config_file e o checkpoint.
#
# É preciso também baixar o arquivo .pth no site do mmdetection e colocar dentro da
# pasta mmdetection/checkpoints. Os arquivos .pth para rede vfnet, por exemplo, podem ser
# encontrados no link abaixo:
# https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/README.md
#
# Dentro do site procure por um link chamado 'model' (podem ter vários, para as várias versões da
# rede que você pode escolher)
MODELS_CONFIG = {
    'vfnet_r50': {
        'config_file': 'configs/vfnet/vfnet_r50_fpn_1x_coco.py',
        'checkpoint' : pasta_checkpoints+'/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth'
    },
    'atss_r50':{
        'config_file': 'configs/atss/atss_r50_fpn_1x_coco.py',
        'checkpoint' : pasta_checkpoints+'/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
    },
    'retinanet_r50':{
        'config_file': 'configs/retinanet/retinanet_r50_fpn_1x_coco.py',
        'checkpoint': pasta_checkpoints+'/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    },
    'faster_r50':{
        'config_file': 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'checkpoint': pasta_checkpoints+'/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    },
    
}

print('Arquiteturas que serão testadas:')
print(MODELS_CONFIG)




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# AJUSTA O ARQUIVO DE CONFIGURAÇÃO DE UMA REDE E AS PASTAS DE
# ENTRA E SAÍDA
#
def setCFG(selected_model,
           data_root,
           classes,
           total_epochs=EPOCAS,
           learning_rate=TAXA_APRENDIZAGEM,
           fold='fold_1'):

  config_file = os.path.join(pasta_mmdetection,MODELS_CONFIG[selected_model]['config_file'])

  from mmdet.apis import set_random_seed
  print('Configuração da rede: ',config_file)
  cfg = Config.fromfile(config_file)

  # Modify dataset type and path
  cfg.data_root = data_root#
  cfg.classes = classes  

  #defining configuration for test dataset
  cfg.data.test.type = cfg.dataset_type
  cfg.data.test.data_root = cfg.data_root
  cfg.data.test.ann_file = 'filesJSON/'+fold+'_test.json'
  cfg.data.test.classes = cfg.classes
  cfg.data.test.img_prefix = 'all/train' # As imagens ficam todos em all/train mesmo 
  # SÃO OS ARQUIVO .JSON QUE FAZEM A DIVISÃO"

  #defining configuration for train dataset
  cfg.data.train.type = cfg.dataset_type
  cfg.data.train.data_root = cfg.data_root
  cfg.data.train.ann_file = 'filesJSON/'+fold+'_train.json'
  cfg.data.train.classes = cfg.classes
  cfg.data.train.img_prefix = 'all/train'# As imagens ficam todos em all/train mesmo 
  # SÃO OS ARQUIVO .JSON QUE FAZEM A DIVISÃO"

  #defining configuration for val dataset
  cfg.data.val.type = cfg.dataset_type
  cfg.data.val.data_root = cfg.data_root
  cfg.data.val.ann_file = 'filesJSON/'+fold+'_val.json'
  cfg.data.val.classes = cfg.classes
  cfg.data.val.img_prefix =  'all/train' # As imagens ficam todos em train mesmo 
  # SÃO OS ARQUIVO .JSON QUE FAZEM A DIVISÃO"
  cfg.data.val.pipeline = cfg.data.train.pipeline
  
  # modify num classes of the model in box head
  if 'roi_head' in cfg.model:
    #cfg.test_cfg.rcnn['score_thr']= 0.51
    if not isinstance(cfg.model.roi_head.bbox_head,list):
      cfg.model.roi_head.bbox_head['num_classes'] = len(cfg.classes)
    else: 
      for i in range(len(cfg.model.roi_head.bbox_head)):
        cfg.model.roi_head.bbox_head[i]['num_classes'] = len(cfg.classes)
  else:
      cfg.model.bbox_head['num_classes'] = len(cfg.classes)
      print('Número de classes: ',cfg.model.bbox_head['num_classes'],'bbox_head:',len(cfg.model.bbox_head))


  

  # We can still use the pre-trained Mask RCNN model though we do not need to
  # use the mask branch
  cfg.load_from =  MODELS_CONFIG[selected_model]['checkpoint']

  # Set up working dir to save files and logs.
  cfg.work_dir = os.path.join(data_root,fold,'MModels/%s'%(selected_model))
  print('Modelos serão salvos aqui: ',cfg.work_dir)
  cfg.total_epochs = total_epochs
#  cfg.runner.max_epochs = total_epochs

  cfg.optimizer.lr = learning_rate 
  #cfg.lr_config.warmup = None
  #cfg.log_config.interval = 100
  cfg.lr_config.policy = 'step'

  # Change the evaluation metric since we use customized dataset.
  cfg.evaluation.metric = 'bbox'
  cfg.evaluation.save_best='auto'
  # We can set the evaluation interval to reduce the evaluation times
  cfg.evaluation.interval = 1
  # We can set the checkpoint saving interval to reduce the storage cost
  #cfg.checkpoint_config.interval = total_epochs/5
  cfg.checkpoint_config.interval = total_epochs # Vai salvar só o último mesmo
  cfg.checkpoint_config.create_symlink=True

  # Set seed thus the results are more reproducible
  cfg.seed = 0
  set_random_seed(0, deterministic=False)
  cfg.gpu_ids = range(1)


  # We can initialize the logger for training and have a look
  # at the final config used for training
  #print(f'Config:\n{cfg.pretty_text}')
  return cfg

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# FUNÇÃO AUXILIAR PARA ESCREVER EM ARQUIVO
#
  
# Vai salar os resultados no arquivo dataset/results.csv
def printToFile(linha='',arquivo='dataset/results.csv',modo='a'):
  original_stdout = sys.stdout # Save a reference to the original standard output
  with open(arquivo, modo) as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(linha)
    sys.stdout = original_stdout # Reset the standard output to its original value



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# FUNÇÃO QUE FAZ O TREINAMENTO DA REDE  
#
def trainModel(cfg):
  cfg.workflow = [('train', 1),('val', 1)]

  torch.backends.cudnn.benchmark = True
  distributed = False

  # Create work_dir
  print('Create workdir:',mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir)))

  # dump config
  cfg.dump(osp.join(cfg.work_dir, osp.basename(selected_model+'.py')))
  # init the logger before other steps
  timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
  log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
  logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

  # init the meta dict to record some important information such as
  # environment info and seed, which will be logged
  meta = dict()
  # log env info
  env_info_dict = collect_env()
  env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
  dash_line = '-' * 60 + '\n'
  logger.info('Environment info:\n' + dash_line + env_info + '\n' +
              dash_line)
  meta['env_info'] = env_info
  meta['config'] = cfg.pretty_text
  # log some basic info
  logger.info(f'Distributed training: {distributed}')
  logger.info(f'Config:\n{cfg.pretty_text}')

  # set random seeds
  meta['seed'] = cfg.seed
  meta['exp_name'] = osp.basename(selected_model+'.py')


  # Build dataset
  datasets = [build_dataset(cfg.data.train,dict(test_mode=False,filter_empty_gt=False))]
  datasets.append(build_dataset(cfg.data.val,dict(test_mode=False,filter_empty_gt=False)))


  datasets[0].CLASSES = cfg.classes
  datasets[1].CLASSES = cfg.classes

  cfg.checkpoint_config.meta = dict(
              mmdet_version=__version__ + get_git_hash()[:7],
              CLASSES=datasets[0].CLASSES)

  # Build the detector
  #model = build_detector(
  #    cfg.model)
  model = build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
  # Add an attribute for visualization convenience
  model.CLASSES = datasets[0].CLASSES

  train_detector(model, datasets, cfg, distributed=False, validate=False,timestamp=timestamp,meta=meta)




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# FUNÇÃO QUE APLICA O MODELO APRENDIDO NOS DADOS DE TESTE
#
def testingModel(cfg=None,typeN='test',models_path=None,show_imgs=False,save_imgs=False,num_model=1,fold='fold_1'):
  
  # build the model from a config file and a checkpoint file
  cfg.data.test.test_mode = True
  torch.backends.cudnn.benchmark = True
  cfg.model.pretrained = None

  modelx = init_detector(cfg, models_path)
  
  if typeN=='test':
    ann_file = cfg.data.test.ann_file
    img_prefix = cfg.data.test.img_prefix
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
  elif typeN=='validation':
    ann_file = cfg.data.val.ann_file
    img_prefix = cfg.data.val.img_prefix  
  elif typeN=='train':
    ann_file = cfg.data.train.ann_file
    img_prefix = cfg.data.train.img_prefix  

  coco_dataset = CocoDataset(ann_file=ann_file, classes=cfg.classes,data_root=cfg.data_root,img_prefix=img_prefix,pipeline=cfg.train_pipeline,filter_empty_gt=False)

  MAX_BOX=1000
  results=[]
  medidos=[]
  preditos=[]
  for i,dt in enumerate(coco_dataset.data_infos):

    print('Arquivo de imagens:',dt['file_name'])

    imagex=None
    imagex=mmcv.imread(os.path.join(coco_dataset.img_prefix,dt['file_name']))
    resultx = inference_detector(modelx, imagex)


    #GT BBOXS VERMELHOS  GroundTruth  
    ann = coco_dataset.get_ann_info(i)
    bboxes = np.insert(ann['bboxes'],4,0.91,axis=1)

    #vis.imshow_gt_det_bboxes(imagex,dict(gt_bboxes=bboxes, gt_labels=np.repeat(1, len(bboxes))), resultx,det_bbox_color=(0,100,0), show=True,score_thr=0.5)

    objetos_medidos=bboxes.shape[0] # Total de objetos marcados manualmente (groundtruth)
    for j in range(min(MAX_BOX, bboxes.shape[0])): 
      left_top = (bboxes[j, 0], bboxes[j, 1])
      right_bottom = (bboxes[j, 2], bboxes[j, 3])
      imagex=cv2.rectangle(imagex, left_top, right_bottom, color_val('red'), thickness=2)
    
    #RESULTADOS BBOXS VERDES Prediction
    bboxes = resultx[0]
    objetos_preditos=bboxes.shape[0] # Total de objetos preditos automaticamente

    medidos.append(objetos_medidos)
    preditos.append(objetos_preditos)

    for j in range(min(MAX_BOX, bboxes.shape[0])):
      if bboxes[j,4] > 0.5:
        left_top = (bboxes[j, 0], bboxes[j, 1])
        right_bottom = (bboxes[j, 2], bboxes[j, 3])
        imagex=cv2.rectangle(imagex, left_top, right_bottom, color_val('green'), thickness=1)

    if show_imgs and i<10:  ## VAI MOSTRAR APENAS 10 IMAGENS PARA NÃO FICAR LENTO!
      cv2.imshow(imagex)
    elif save_imgs:
      save_path = cfg.data_root+'/prediction_'+selected_model
      save_path = os.path.join(cfg.data_root,save_path)
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      img_path = os.path.join(save_path ,dt['file_name'])
      

      print('Salvando Imagens com Resultados Em:',img_path)
      print("Salvou?: ",cv2.imwrite(img_path,imagex))

    results.append(resultx)
    printToFile(str(num_model)+'_'+selected_model + ','+fold+','+str(objetos_medidos)+','+str(objetos_preditos),'dataset/counting.csv','a')
    
    
  eval_results = coco_dataset.evaluate(results, classwise=True)
  print('Resultados do comando coco_dataset.evaluate:')
  print(eval_results)
  #print(selected_model,'\t',eval_results['bbox_mAP_50'])
  #string_results = selected_model+'\t'+str(eval_results['bbox_mAP_50'])

  string_results = '0,0,0,0'

  try:
    mAP50=eval_results['bbox_mAP_50']
    MAE=mean_absolute_error(medidos,preditos)
    RMSE=math.sqrt(mean_squared_error(medidos,preditos))
    r=np.corrcoef(medidos,preditos)[0,1]
    string_results = str(mAP50)+','+str(MAE)+','+str(RMSE)+','+str(r)
  except:
    print('ERRO ... DEU PROBLEMA NA HORA DE CALCULAR ALGUMA MÉTRICA')

  return string_results
  
  




#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# RODA O TREINAMENTO PARA TODOS AS REDES, VARIANDO O TAMANHO DA BBOX E
# USANDO AS 5 DOBRAS 
#

REDES=[k for k in MODELS_CONFIG.keys()]

if(not APENAS_TESTA):
  for selected_model in REDES:
    for f in np.arange(1,DOBRAS+1):
      print('------------------------------------------------------')
      print('-- RODANDO COM A REDE ',selected_model,' NA DOBRA ',f)
      print('------------------------------------------------------')
      fold = 'fold_'+str(f)
      cfg = setCFG(selected_model=selected_model,data_root=pasta_dataset,classes=('Corn',),fold=fold)
      trainModel(cfg)




    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#
# RODA NO CONJUNTO DE TESTE
#


print('======================================================')
print(' INICIANDO TESTE - INICIANDO TESTE - INICIANDO TESTE')
print('======================================================')

printToFile('ml,fold,groundtruth,predicted','dataset/counting.csv','w')

printToFile('ml,fold,mAP50,MAE,RMSE,r','dataset/results.csv','w')
i=1
for selected_model in REDES:
    for f in np.arange(1,DOBRAS+1):
      print('------------------------------------------------------')
      print('-- TESTANDO A REDE ',selected_model,' NA DOBRA ',f)
      print('------------------------------------------------------')

      fold = 'fold_'+str(f)
      cfg = setCFG(selected_model=selected_model,data_root=pasta_dataset,classes=('Corn',),fold=fold)

      pth = os.path.join(cfg.data_root,(fold+'/MModels/%s/latest.pth'%(selected_model)))
      print('Usando o modelo aprendido: ',pth)
      resAP50 = testingModel(cfg=cfg,models_path=pth,show_imgs=False,save_imgs=True,num_model=i,fold=fold)
      printToFile(str(i)+'_'+selected_model + ','+fold+','+resAP50,'dataset/results.csv','a')
    i=i+1
  

