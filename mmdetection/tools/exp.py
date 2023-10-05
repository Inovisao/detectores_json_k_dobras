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

# Lembre que a ordem das classes deve ser a mesma que está no arquivo .json

DATA_ROOT          = '/home/junior/Expjr/detectores_json_k_dobras/dataset/'
CLASSES            = ('pig',)
#CLASSES            = ('Swollen-Ears','Good_Ears',)
BATCH_SIZE         = 2
MAX_EPOCHS         = 5
LEARNING_RATE      = 0.0001
OPTMIZER           = 'SGD'
THRESHOLD          = 0.2
THRESHOLD_CLASSIFY = 0.3
DEVICE             = 'cuda:0'
DATASET_TYPE       = 'CocoDataset'
MAX_IMG_SAVE       = 10

class Service(object):

    def __init__(self, config, train, val, test, launcher='none'):
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
            self.change_cfg(train=train,test=test,val=val, config=config)

            #print(self.cfg)
            #exit(1)
              # build the runner from config
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

            
    def create_path(self, *args):
        try:
            path = ''
            for arg in args:
                path = os.path.join(path, arg)
            return path
        except ValueError as error:
            print(error)

    def colors(self,size):
        return [(random.randint(a=0,b=255),random.randint(a=0,b=255), random.randint(a=0,b=255)) for _ in range(size)]
    
    def change_cfg(self, train, test, val, config):
        try:
            self.cfg.data_root                    = DATA_ROOT
            self.cfg.train_dataloader.batch_size  = BATCH_SIZE
            self.cfg.total_epochs                 = MAX_EPOCHS
            self.cfg.optim_wrapper.optimizer.lr   = LEARNING_RATE
            self.cfg.optim_wrapper.optimizer.type = OPTMIZER
            
            if 'data_root' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.data_root              = DATA_ROOT
            if 'dataset' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.dataset['metainfo']    = dict(classes=CLASSES)
                #self.cfg.train_dataloader.dataset.dataset.CLASSES        = CLASSES
                if 'ann_file' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.ann_file   = self.create_path('filesJSON',str(train))
                if 'data_prefix' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.data_prefix= dict(img='all/train/')
            else:
                self.cfg.train_dataloader.dataset['metainfo']     = dict(classes=CLASSES)
                if 'ann_file' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.ann_file    = self.create_path('filesJSON',str(train))
                if 'data_prefix' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.data_prefix = dict(img='all/train/')
            #if 'metainfo' not in self.cfg.train_dataloader.dataset:
            #    self.cfg.train_dataloader.dataset['metainfo'] = dict(classes=CLASSES)
            #else:
            #    self.cfg.train_dataloader.dataset['metainfo'] = dict(classes=CLASSES)

            self.cfg.val_dataloader.batch_size = BATCH_SIZE
            #self.cfg.val_dataloader.CLASSES  = CLASSES
            if 'data_root' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.ann_file = self.create_path('filesJSON',str(val))
            if 'data_prefix' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset['metainfo'] = dict(classes=CLASSES)

            self.cfg.test_dataloader.batch_size = BATCH_SIZE
            #self.cfg.test_dataloader.CLASSES   = CLASSES
            if 'data_root' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.ann_file = self.create_path('filesJSON',str(test))
            if 'data_prefix' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset['metainfo'] = dict(classes=CLASSES)

            if 'bbox_head' in self.cfg.model:
                self.cfg.model.bbox_head.num_classes=len(CLASSES)
                if 'mask_head' in self.cfg.model.bbox_head:
                    self.cfg.model.bbox_head.mask_head.num_classes=len(CLASSES)

            name_work_dir = self.create_path('work_dirs' , config.split('.py')[0])
            self.cfg.val_evaluator.ann_file   = self.create_path(DATA_ROOT , 'filesJSON' , str(val))
            self.cfg.test_evaluator.ann_file  = self.create_path(DATA_ROOT , 'filesJSON' , str(test))
            self.cfg.max_epochs               = MAX_EPOCHS
            self.cfg.work_dir                 = self.create_path(os.getcwd(),name_work_dir)
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

    def download_models(self):
        try:
            configs =[
                    #'rtmdet_tiny_8xb32-300e_coco',
                    #'cornernet_hourglass104_8xb6-210e-mstest_coco',
                    'ssd300_coco',
                    #'paa_r50_fpn_1x_coco',
                    #'tridentnet_r50-caffe_1x_coco',
                    #'detr_r50_8xb2-150e_coco',# https://github.com/open-mmlab/mmdetection/tree/main/configs/detr
                    
                ]
            for config in configs:
                print(config)
                os.system('mim download mmdet --config '+config+' --dest checkpoints/')
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
    
    def iou(self, bb1, bb2):
        """This method is responsable for compute Area Over Union
        bb1: list(objects)
            Define the list of objects(detection)
        bb2: list(objects)
            Define the lista of objects(detection)
        """
        try:
            assert bb1[0] < (bb1[0]+bb1[2])
            assert bb1[1] < (bb1[1]+bb1[3])
            assert bb2[0] < (bb2[0]+bb2[2])
            assert bb2[1] < (bb2[1]+ bb2[3])

            x_left   = max(bb1[0], bb2[0])
            y_top    = max(bb1[1], bb2[1])
            x_right  = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
            y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bb1_area = ((bb1[0]+bb1[2]) - bb1[0]) * ((bb1[1]+bb1[3]) - bb1[1])
            bb2_area = ((bb2[0]+bb2[2]) - bb2[0]) * ((bb2[1]+bb2[3]) - bb2[1])
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            assert iou >= 0.0
            assert iou <= 1.0
            return iou
        except ValueError as error:
            print(error)

    def processing_predicts(self, scores, bboxes, labels):
        """
            Este método é responsável por receber dados de predição deixar somente os boxes que tenham scores maiores de THRESHOLD
        """
        try:
            bboxes_pred = []
            #Getting only boxes with scores more than threshold
            for i, score in enumerate(scores):
                if score >= THRESHOLD_CLASSIFY:
                    bboxes_pred.append({'class':labels[i], 'score':score, 'bbox':bboxes[i]})
            return bboxes_pred
        except ValueError as error:
            print(error)
            
    def select_more_close(self, bbox_ground, bboxes_pred):
        try:
            val_iou = THRESHOLD
            bbox    = None
            index   = 0
            #Lets go found the minor distance
            for i,bbox_pred in enumerate(bboxes_pred):
                value = self.iou( bbox_ground.get('bbox') , bbox_pred.get('bbox'))
                if value >= val_iou:
                    bbox    = bbox_pred
                    val_iou = value
                    index   = i
            print('Distance..:',val_iou)
            return bbox, index 
        except ValueError as error:
            print(error)
            
    def draw_rectangle(self, elements, img, color=(255,0,0),default=False, classe=None):
        try:
            bbox  = [int(x) for x in elements.get('bbox')]
            #print(bbox)
            if default == True:
                bbox2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
            else:
                bbox2 = [bbox[2],  bbox[3]]

            if classe is not None:
                position = (bbox[0]+5, bbox[1]+15)
                img = cv2.putText(img,classe, position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1, cv2.LINE_AA)

            return cv2.rectangle(img, bbox[:2], bbox2, color, thickness=2)
        except ValueError as error:
            print(error)

    def testing(self, path, checkpoint, config='rtmdet_tiny_8xb32-300e_coco.py'):
        try:
            
            files     = self.get_files(path)
            test      = files.get('test')
            model     = init_detector(config, checkpoint, device=DEVICE)
            tp_all    = 0              # This variable is responsable for counting true positive from all images
            fp_all    = 0              # This variable is responsable for counting false positive from all images
            all_imgs  = 0              # This variable is responsable for counting all images 

            for key in test:           # All folders 
                p_test      = os.path.join(os.path.join(DATA_ROOT , 'filesJSON'), test.get(key))
                coco        = COCO(p_test)
                image_ids   = coco.getImgIds()
                count_image = 0
                # Um problema de utilizar o config é que não necessariamente estará no diretório works_dir
                # Lembrar de mudar o local_save
                name  = (test.get(key)).split('test')[0]
                local_save = os.path.join(os.path.dirname(checkpoint), name)
                print(local_save) 
                if not os.path.exists( local_save ):
                    os.mkdir( local_save )
                
                for id in image_ids:       # All images in k-folder
                    image_info = coco.loadImgs(id)
                    print(dir(coco))
                    #print(coco.cats)
                    
                    # Segundo o mmdetection a ordem das classes no .json deve ser a mesma que a classes
                    cl = {}
                    for d in coco.cats.values():
                        na = d.get('name')
                        cl[ d.get('id') ] = CLASSES.index( na )
                    
                    name_img       = image_info[0].get('file_name')
                    img            = mmcv.imread( os.path.join(os.path.join(DATA_ROOT,'all/train') , name_img))
                    annotations    = coco.loadAnns( coco.getAnnIds(imgIds=id) )
                    bboxes_gt      = []
                    tp_img         = 0     # This variable is responsable for counting true positive from one image
                    fp_img         = 0     # This variable is responsable for counting false positive from one image
                    all_img        = 0     # This variable is responsable for counting all images boxes
                    print('-'*10,'Processing [',name_img, '] with..: ',str(len(annotations)),' annotations.\n')
                    
                    all_img = len(annotations)
                    for annotation in annotations:
                        #print(annotation.get('category_id'))
                        #This line abouve is responsable for show results
                        img  = self.draw_rectangle(annotation, img, default=True)
                        bboxes_gt.append({'class':cl.get( annotation.get('category_id')), 'score':0, 'bbox':annotation.get('bbox')})

                    #Realize the prediction of image
                    result = inference_detector(model, img)

                   
                    #Getting all boxes with value equal or more than THRESHOLD of Classification. 
                    bboxes_pred = self.processing_predicts(result.pred_instances.scores.cpu().numpy(), result.pred_instances.bboxes.cpu().numpy(), result.pred_instances.labels.cpu().numpy())
                    #print(bboxes_pred)

                    #This lines are responsable for walking for all bbox predicted
                    for bbox_gt in bboxes_gt:
                        bbox, index = self.select_more_close(bbox_gt, bboxes_pred)
                        if bbox:
                            if bbox.get('class') == bbox_gt.get('class'):
                                print('True Positive')
                                tp_img += 1
                                img  = self.draw_rectangle(bbox, img ,color=(0,255,0),classe=CLASSES[int(bbox.get('class'))])
                                del bboxes_pred[index] #This line remove the boxes positive, because its not be used
                            else:
                                print('The box is in same local. But class is different.')
                                img  = self.draw_rectangle(bbox, img ,color=(0,255,255))
                        else:
                            print('False Positive')
                            fp_img += 1
                            
                    for pred in bboxes_pred:
                        img  = self.draw_rectangle(pred, img ,color=(0,0,255))

                    img = cv2.putText(img,'TP: '+str(tp_img),  (10,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1, cv2.LINE_AA)
                    img = cv2.putText(img,'GT: '+str(all_img), (10,35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1, cv2.LINE_AA)

                    #This line is responsable for saving in k-folder-test
                    if count_image < MAX_IMG_SAVE:
                        cv2.imwrite( os.path.join(local_save, name_img),img)
                    count_image += 1
                    
                    tp_all   += tp_img
                    fp_all   += fp_img
                    all_imgs += all_img
                print('TP..:%d, FP..:%d, ALL..:%d in each k-folder' % (tp_all, fp_all, all_imgs))
                print('This code is executing only image. For execute all images to remove line 258')
                #exit(1)
                
        except ValueError as error:
            print(error)

    def running(self, path, config='rtmdet_tiny_8xb32-300e_coco.py'):
        try:
            files = self.get_files(path)
            train = files.get('train')
            test  = files.get('test')
            val   = files.get('val')
            for i,key in enumerate(train):
                p_train = train.get(key)
                p_val   = val.get(key)
                p_test  = test.get(key)

                print(p_train)
                service = Service(config, train=p_train, val=p_val,test=p_test)
                #if i == 2:
                #    print('Remember of eraser lines 303-305 after testing this code!!!')
                #    exit(1)
                #print('To Processing all. You must eraser the line 294.')
                #exit(1)
        except ValueError as error:
            print(error)


# This is my references: https://gist.github.com/interactivetech/c2913317603b79c02ff49fa9824f1104

if __name__ == '__main__':

    train = Training()
    #train.download_models()
    
    #train.running('../dataset/filesJSON', config='checkpoints/ssd300_coco.py')
    #train.testing('../dataset/filesJSON', config='work_dirs/ssd300_coco/ssd300_coco.py' ,checkpoint='work_dirs/ssd300_coco/best_coco_bbox_mAP_epoch_10.pth')

    train.testing_('../dataset/filesJSON', config='work_dirs/ssd300_coco/ssd300_coco.py' ,checkpoint='work_dirs/ssd300_coco/best_coco_bbox_mAP_epoch_10.pth')


    #train.running('../dataset/filesJSON', config='checkpoints/paa_r50_fpn_1x_coco.py')
    #train.testing('../dataset/filesJSON', config='work_dirs/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco.py' ,checkpoint='work_dirs/rtmdet_tiny_8xb32-300e_coco/best_coco_bbox_mAP_epoch_20.pth')
    

    print('ok')
