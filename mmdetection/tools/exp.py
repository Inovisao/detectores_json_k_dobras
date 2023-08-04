# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
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


DATA_ROOT          = '/home/jr/development/inovisao/dev/detectores_json_k_dobras/dataset'
CLASSES            = dict(classes=('pig',))
BATCH_SIZE         = 5
MAX_EPOCHS         = 5
LEARNING_RATE      = 0.0003
OPTMIZER           = 'SGD'
THRESHOLD          = 0.2
THRESHOLD_CLASSIFY = 0.2

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

        except ValueError as error:
            print(error)

    def change_cfg(self, train, test, val, config):
        try:
            self.cfg.data_root = DATA_ROOT
            self.cfg.train_dataloader.batch_size = BATCH_SIZE
            self.cfg.total_epochs = MAX_EPOCHS
            self.cfg.optim_wrapper.optimizer.lr   = LEARNING_RATE
            self.cfg.optim_wrapper.optimizer.type = OPTMIZER
            
            if 'data_root' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.data_root = DATA_ROOT
            if 'dataset' in self.cfg.train_dataloader.dataset:
                self.cfg.train_dataloader.dataset.dataset['metainfo']=dict(CLASSES)
                
                if 'ann_file' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.ann_file = 'filesJSON/'+str(train)
                if 'data_prefix' in self.cfg.train_dataloader.dataset.dataset:
                    self.cfg.train_dataloader.dataset.dataset.data_prefix = dict(img='all/train/')
            else:
                self.cfg.train_dataloader.dataset['metainfo']=dict(CLASSES)
                if 'ann_file' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.ann_file = 'filesJSON/'+str(train)
                if 'data_prefix' in self.cfg.train_dataloader.dataset:
                    self.cfg.train_dataloader.dataset.data_prefix = dict(img='all/train/')
            #if 'metainfo' not in self.cfg.train_dataloader.dataset:
            #    self.cfg.train_dataloader.dataset['metainfo'] = dict(classes=('pig',))
            #else:
            #    self.cfg.train_dataloader.dataset['metainfo'] = dict(classes=('pig',))

            self.cfg.val_dataloader.batch_size= BATCH_SIZE
            if 'data_root' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.ann_file = 'filesJSON/'+str(val)
            if 'data_prefix' in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.val_dataloader.dataset:
                self.cfg.val_dataloader.dataset['metainfo'] = dict(CLASSES)

            self.cfg.test_dataloader.batch_size= BATCH_SIZE
            if 'data_root' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_root = DATA_ROOT
            if 'ann_file' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.ann_file = 'filesJSON/'+str(test)
            if 'data_prefix' in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset.data_prefix = dict(img='all/train/')
            if 'metainfo' not in self.cfg.test_dataloader.dataset:
                self.cfg.test_dataloader.dataset['metainfo'] = dict(CLASSES)

            name_work_dir = os.path.join('work_dirs' , config.split('.py')[0])
            self.cfg.val_evaluator.ann_file  = DATA_ROOT + '/filesJSON/'+str(val)
            self.cfg.test_evaluator.ann_file = DATA_ROOT + '/filesJSON/'+str(test)
            self.cfg.max_epochs  = MAX_EPOCHS
            self.cfg.work_dir    = os.path.join(os.getcwd(),name_work_dir)
            self.cfg.num_classes = len(CLASSES)
            self.cfg.train_batch_size_per_gpu = 2
            self.cfg.dataset_type = 'CocoDataset'

        except ValueError as error:
            print(error)

class Training(object):

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
            
    def draw_rectangle(self, elements, img, color=(255,0,0)):
        try:
            bbox  = [int(x) for x in elements.get('bbox')]
            print(bbox)
            bbox2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
            return cv2.rectangle(img, bbox[:2], bbox2, color, thickness=1)
        except ValueError as error:
            print(error)
            
    def testing(self, path, checkpoint, config='rtmdet_tiny_8xb32-300e_coco.py'):
        try:
            files = self.get_files(path)
            test  = files.get('test')
            model = init_detector(config, checkpoint, device='cuda:0')
            tp_all    = 0              # This variable is responsable for counting true positive from all images
            fp_all    = 0              # This variable is responsable for counting false positive from all images
            all_imgs  = 0              # This variable is responsable for counting all images 

            for key in test:           # All folders 
                p_test    = DATA_ROOT + '/filesJSON/'+ test.get(key)
                coco      = COCO(p_test)
                image_ids = coco.getImgIds()
                print(p_test)
                
                for id in image_ids:       # All images in k-folder
                    image_info = coco.loadImgs(id)
                    #print(image_info)
                    name_img       = image_info[0].get('file_name')
                    img            = mmcv.imread(DATA_ROOT+'/all/train/'+name_img)
                    annotations    = coco.loadAnns( coco.getAnnIds(imgIds=id) )
                    bboxes_gt      = []
                    tp_img         = 0     # This variable is responsable for counting true positive from one image
                    fp_img         = 0     # This variable is responsable for counting false positive from one image
                    all_img        = 0     # This variable is responsable for counting all images boxes
                    print('-'*10,'Processing [',name_img, '] with..: ',str(len(annotations)),' annotations.\n')
                    
                    all_img = len(annotations)
                    for annotation in annotations:
                        #print(annotation.get('category_id'))
                        #print(annotation.get('bbox'))
                        #This line abouve is responsable for show results
                        img  = self.draw_rectangle(annotation, img)
                        bboxes_gt.append({'class':annotation.get('category_id'), 'score':0, 'bbox':annotation.get('bbox')})

                    #Realize the prediction of image
                    result = inference_detector(model, img)
                    #Lets getting all boxes with value equal or more than THRESHOLD of Classification. 
                    bboxes_pred = self.processing_predicts(result.pred_instances.scores.cpu().numpy(), result.pred_instances.bboxes.cpu().numpy(), result.pred_instances.labels.cpu().numpy())
                    print(bboxes_pred)

                    #This line is responsable for walking for all bbox predicted
                    for bbox_gt in bboxes_gt:
                        bbox, index = self.select_more_close(bbox_gt, bboxes_pred)
                        if bbox:
                            if bbox.get('class') == bbox_gt.get('class'):
                                print('True Positive')
                                tp_img += 1
                                img  = self.draw_rectangle(bbox, img ,color=(0,255,0))
                                print(' **** Hit in this moments ***')
                                exit(1)
                            else:
                                print('The box is in same local. But class is different.')
                                img  = self.draw_rectangle(bbox, img ,color=(0,255,255))
                            del bboxes_pred[index] #This line remove the boxes positive, because its not be used
                        else:
                            print('False Positive')
                            fp_img += 1
                            
                    for pred in bboxes_pred:
                        img  = self.draw_rectangle(pred, img ,color=(0,0,255))

                    cv2.imwrite('res/'+name_img,img)
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
            for key in train:
                p_train = train.get(key)
                p_val   = val.get(key)
                p_test  = test.get(key)

                print(p_train)
                service = Service(config, train=p_train, val=p_val,test=p_test)
        except ValueError as error:
            print(error)


if __name__ == '__main__':

    train = Training()
    train.running('../dataset/filesJSON')
    #train.testing('../dataset/filesJSON', config='work_dirs/ssd300_coco/ssd300_coco.py' ,checkpoint='work_dirs/ssd300_coco/epoch_24.pth')
    print('ok')
