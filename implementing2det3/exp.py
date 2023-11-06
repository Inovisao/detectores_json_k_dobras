# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import torch
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmdet.apis import init_detector, inference_detector


from pycocotools.coco import COCO
import mmcv
import cv2
import random

from arguments import Arguments
from experimenter import Testing

# Lembre que a ordem das classes deve ser a mesma que está no arquivo .json
DATA_ROOT          = '/home/junior/Expjr/SUINDETEC'
CLASSES            = ('pig',)
#CLASSES            = ('Swollen-Ears','Good_Ears',)
BATCH_SIZE         = 2
MAX_EPOCHS         = 3
LEARNING_RATE      = 0.0001
OPTMIZER           = 'SGD'
THRESHOLD          = 0.2
THRESHOLD_CLASSIFY = 0.3
DEVICE             = 'cuda:0'
DATASET_TYPE       = 'CocoDataset'
MAX_IMG_SAVE       = 5

class Service(object):

    def __init__(self, config, train, val, test, local_work_dir='work_dirs',launcher='none'):
        try:
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = str(0)

            setup_cache_size_limit_of_dynamo()
            self.cfg          = Config.fromfile(config)
            self.cfg.launcher = launcher

            optim_wrapper = self.cfg.optim_wrapper.type
            assert optim_wrapper == 'OptimWrapper', ('`--amp` is only supported when the optimizer wrapper type is 'f'`OptimWrapper` but got {optim_wrapper}.')
            self.cfg.optim_wrapper.type = 'AmpOptimWrapper'
            self.cfg.optim_wrapper.loss_scale = 'dynamic'
            self.change_cfg(train=train,test=test,val=val, config=config, local_work_dir=local_work_dir)

            
            if 'runner_type' not in self.cfg:# build the default runner
                runner = Runner.from_cfg(self.cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(self.cfg)
                
            runner = Runner.from_cfg(self.cfg)
            # start training
            runner.train()

            # start testing
            runner.val()
        except ValueError as error:
            print(error)

            
    def colors(self,size):
        return [(random.randint(a=0,b=255),random.randint(a=0,b=255), random.randint(a=0,b=255)) for _ in range(size)]

    
    def change_cfg(self, train, test, val, config,local_work_dir):
        try:
            self.cfg.data_root                    = DATA_ROOT
            self.cfg.train_dataloader.batch_size  = BATCH_SIZE
            self.cfg.total_epochs                 = MAX_EPOCHS
            self.cfg.optim_wrapper.optimizer.lr   = LEARNING_RATE
            self.cfg.optim_wrapper.optimizer.type = OPTMIZER
            self.name_work_dir                    = config.split('.py')[0]
            
            if 'data_root' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.data_root              = DATA_ROOT
            if 'dataset' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.dataset['metainfo']    = dict(classes=CLASSES)
                if 'ann_file' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.ann_file   = os.path.join('filesJSON',str(train))
                    self.cfg.train_dataloader.dataset.dataset.data_root  = DATA_ROOT
                if 'data_prefix' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.data_prefix= dict(img='all/train/')
            else:
                self.cfg.train_dataloader.dataset['metainfo']     = dict(classes=CLASSES)
                if 'ann_file' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.ann_file    = os.path.join('filesJSON',str(train))
                if 'data_prefix' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.data_prefix = dict(img='all/train/')
        

            self.cfg.val_dataloader.batch_size = BATCH_SIZE
            if 'data_root' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.ann_file = os.path.join('filesJSON',str(val))
            if 'data_prefix' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset['metainfo'] = dict(classes=CLASSES)

            self.cfg.test_dataloader.batch_size = BATCH_SIZE
            if 'data_root' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.ann_file = os.path.join('filesJSON',str(test))
            if 'data_prefix' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset['metainfo'] = dict(classes=CLASSES)

            if 'bbox_head' in self.cfg.model:
                self.cfg.model.bbox_head.num_classes=len(CLASSES)
                if 'mask_head' in self.cfg.model.bbox_head:
                    self.cfg.model.bbox_head.mask_head.num_classes=len(CLASSES)

            
            self.cfg.val_evaluator.ann_file   = os.path.join(DATA_ROOT , 'filesJSON' , str(val))
            self.cfg.test_evaluator.ann_file  = os.path.join(DATA_ROOT , 'filesJSON' , str(test))
            self.cfg.max_epochs               = MAX_EPOCHS
            self.cfg.work_dir                 = os.path.join(os.getcwd(),self.name_work_dir)
            self.cfg.num_classes              = len(CLASSES)
            self.cfg.train_batch_size_per_gpu = 2
            self.cfg.dataset_type             = DATASET_TYPE
            self.cfg['train_cfg']             = dict(type='EpochBasedTrainLoop',max_epochs=MAX_EPOCHS,val_interval=MAX_EPOCHS )
            self.cfg['val_cfg']               = dict(type='ValLoop')
            self.cfg['param_scheduler']       = [dict(type='LinearLR', start_factor=0.001,by_epoch=False,begin=0,end=5), dict(type='MultiStepLR', by_epoch=True,  begin=0,   end=12,  milestones=[8, 11],  gamma=0.1)  ]
            self.cfg['default_hooks']         = dict(checkpoint=dict(type='CheckpointHook',save_best='auto'))
            self.cfg['metainfo']              = {'classes': CLASSES,'palette': self.colors(size=len(CLASSES))}
            self.cfg.CLASSES                  = CLASSES
        except ValueError as error:
            print(error)

class Training(object):

    def download_models(self, local='checkpoints/',who='ssd300_coco'):
        try:
            configs =[
                    'rtmdet_tiny_8xb32-300e_coco',
                    'cornernet_hourglass104_8xb6-210e-mstest_coco',
                    'ssd300_coco',
                    'paa_r50_fpn_1x_coco',
                    'tridentnet_r50-caffe_1x_coco',
                    #'detr_r50_8xb2-150e_coco',# https://github.com/open-mmlab/mmdetection/tree/main/configs/detr
                    
                ]
            configs = configs if who == 'all' else [who]
            for config in configs:
                print(config)
                os.system('mim download mmdet --config '+config+' --dest '+local)
        except ValueError as error:
            print(error)
    
    def get_files(self,path):
        files = os.listdir(path)
        names = {'train':{},'val':{},'test':{}}
        for file in files:
            for key in names.keys():
                if file.replace(key,'') != file:
                    dic = names.get(key)
                    n   = file.split('_')
                    dic.update({n[1]:file})
                    names[key] = dic
        return names

    def testing(self, test_files,checkpoints, config, local_work_dir, prefix, name):
        try:
            model = init_detector(config, checkpoints, device=torch.device(DEVICE))
            for p_test in test_files: 
                t = Testing(
                    model        = model,
                    results_p    = local_work_dir,
                    prefix       = os.path.join(prefix,'all/train'),
                    path_json    = os.path.join(prefix,'filesJSON',p_test),
                    technique    = name,
                )
                t.running()

        except ValueError as error:
            print(error)

    def running(self, path_dataset_json,checkpoints,local_work_dir, config='rtmdet_tiny_8xb32-300e_coco.py'):
        try:
            files = self.get_files(path_dataset_json)
            path_dataset_json = path_dataset_json[:-1] if path_dataset_json[-1] == os.sep else path_dataset_json 
            prefix = path_dataset_json.replace(path_dataset_json.split(os.sep)[-1],'')
            
            train = files.get('train')
            test  = files.get('test')
            val   = files.get('val')
            
            for i,key in enumerate(train):
                p_train = train.get(key)
                p_val   = val.get(key)
                p_test  = test.get(key)
                service = Service(config, train=p_train, val=p_val,test=p_test, local_work_dir=checkpoints)
                
                #name         = service.name_work_dir.split(os.sep)[-1]
                #config_train = os.path.join(service.name_work_dir,name+'.py')
                #model_train  = os.path.join(service.name_work_dir,'best_coco_bbox_mAP_epoch_'+str(MAX_EPOCHS)+'.pth')

               
               # self.testing(list([p_test]),checkpoints=model_train, config=config_train, local_work_dir=local_work_dir, prefix=prefix, name=name.split('_')[0])

        except ValueError as error:
            print(error)


# This is my references: https://gist.github.com/interactivetech/c2913317603b79c02ff49fa9824f1104

if __name__ == '__main__':

    """Make testing example:

        python exp.py  -t 0 -p ../checkpoints/ssd300_coco/best_coco_bbox_mAP_epoch_10.pth -m ../checkpoints/ssd300_coco/ssd300_coco.py 

    """
    train  = Training()
    values = {
            
            'b':{
                'name':'download',
                'default':None,
                'help': 'This option is responsable for download of model for training. all for download all'
            },

            't':{
                'name': 'testing',
                'default': None,
                'help': 'This option is responsable for execute only testing'
            },

            'r':{
                'name':'local',
                'default':'.',
                'help': 'This option is responsable define local where saving all operations'
            },

            'm':{
                'name':'model',
                'default':'../checkpoints/ssd300_coco.py',
                'help': 'This option is responsable for download of model for training'
            },

            'd':{
                'name':'dataset',
                'default':'../../SUINDETEC/filesJSON/',
                'help': 'This option is responsable define local os dataset'
            },

            'c':{
                'name':'checkpoints',
                'default':'../checkpoints',
                'help': 'This option is responsable define local checkpoints'
            },

            'p':{
                'name':'.pth',
                'default':'.',
                'help': 'This option is responsable define local of model trainning'
            },

        }
    arguments = Arguments(values)
    arg   = arguments.get()  # Get the parsed arguments.
    local = arg['local'] if arg['local'] is not None else os.getcwd()

    
    if arg['download'] is not None:
        if not os.path.exists(arg['checkpoints']):
            os.mkdir(arg['checkpoints'])
        train.download_models(local=arg['checkpoints'],who=arg['download'])

    elif arg['testing'] is not None:
        path_dataset_json = arg['dataset']
        files             = train.get_files(path_dataset_json)
        path_dataset_json = path_dataset_json[:-1] if path_dataset_json[-1] == os.sep else path_dataset_json 
        prefix            = path_dataset_json.replace(path_dataset_json.split(os.sep)[-1],'')
        tests_f           = files.get('test')

        train.testing(list(tests_f.values()),checkpoints=arg['.pth'], config=arg['model'], local_work_dir=arg['local'], prefix=prefix, name=(arg['.pth']).split(os.sep)[-2])
    
    else:
        
        train.running(path_dataset_json=arg['dataset'],checkpoints=arg['checkpoints'],local_work_dir=arg['local'],config=arg['model'])
    print('ok')
