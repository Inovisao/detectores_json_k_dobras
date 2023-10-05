import torch
from pycocotools.coco import COCO
import os
import torchmetrics
import cv2
import numpy as np

#from mmdet.apis import init_detector, inference_detector
#from torchmetrics.detection import MeanAveragePrecision
#from torchmetrics.functional.detection import intersection_over_union

class ExtractCoco(object):

    def __init__(self, path):
        try:
            assert os.path.exists(path), 'File ['+path+'] not exists!'
            self.coco = COCO(path)
            self.dic_labels = {}
            for element in self.coco.cats.values():
                self.dic_labels[int(element['id'])] = element['name']
        except ValueError as error:
            print(error)
    
    def extract_dic(self):
        try:
            imgs = []
            for key in self.coco.imgs.keys():
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=key))
                self.size          = [self.coco.imgs[key]['width'], self.coco.imgs[key]['height']]
                boxes, labels, areas, iscrowd, masks,image_id = self.getAnnotations(annotations)
                target             = {}
                target['boxes']    = boxes
                target['labels']   = labels
                target['image_id'] = image_id
                target['area']     = areas
                target['iscrowd']  = iscrowd
                imgs.append( [self.coco.imgs[key]['file_name'], target, self.size] )
            return imgs
        except ValueError as error:
            print(error)
    
    def normalize_box(self, box):
        try:
            return [ x/self.size[0] if i==0 or i==2 else x/self.size[1] for i,x in enumerate(box)]
        except ValueError as error:
            print(error)

    def getAnnotations(self, annotations, normalize=False):
        try:
            boxes  = []
            labels = []
            areas  = []
            
            for annotation in annotations:
                # Getting all boxes
                box = [float(val) for val in annotation.get('bbox')]
                #box[2] += box[0]
                #box[3] += box[1]
                if normalize:
                    box = self.normalize_box(box=box)
                boxes.append( box )
                # Getting all labels
                labels.append( annotation.get('category_id') )
                # Getting all areas
                areas.append( annotation.get('area') )
            boxes    = torch.FloatTensor(boxes)
            labels   = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.as_tensor([annotations[0].get('image_id')],dtype=torch.int64)
            areas    = torch.Tensor(areas)
            iscrowd  = torch.zeros((len(annotations),), dtype=torch.uint8)
            masks    = None
            return boxes, labels, areas, iscrowd, masks,image_id
        except ValueError as error:
            print(error)

class Boxes(object):

    def draw_text(self, img, text, position, color=(0,255,0)):
        try:
            img = cv2.putText(img, text,position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color , 1, cv2.LINE_AA)
            return img
        except ValueError as error:
            print(error)

    def draw_rectangle(self, bboxes, img, color=(0,255,0), dim=(200,200), classes=None, normalize=False):
        try:
            for bbox in bboxes:
                prob = None
                if len(bbox) > 1:
                    if len(bbox) > 2:
                        prob = bbox[2]
                    index = bbox[1]
                    bbox  = bbox[0]
                    
                if normalize:
                    bbox = [int(x * dim[0]) if i == 0 or i == 2 else int(x*dim[1])  for i,x in enumerate(bbox) ]
                else:
                    bbox = [int(x) for x in bbox]

                point_1 = (bbox[0], bbox[1])
                point_2 = (bbox[2]+bbox[0], bbox[3]+bbox[1])
                img = cv2.rectangle(img, point_1, point_2, color, thickness=1)
                if classes is not None:
                    position = (bbox[0], bbox[1]+15)
                    msg      = classes[int(index)]
                    if prob is not None:
                        msg += ':'+str(prob) 
                    img      = self.draw_text(img,msg,position,color)
            return img
        except ValueError as error:
            print(error)
            
    def save_file(self,data, local):
        try:
            file_exists = False if os.path.exists(local) else True
            with open(local, 'a') as file:
                headers = list(data.keys())
                headers.sort()
                if file_exists:
                    line = ','.join(headers)
                    file.write(line+'\n')
                line = [str(data[key]) for key in headers]
                line = ','.join(line)
                file.write(line+'\n')
                file.close()

        except ValueError as error:
            print(error)

class Testing(object):

    def __init__(self, model,path,prefix, device='cuda:0'):
        try:
            self.model = model
            assert os.path.exists(path), 'This file not exists!'
            self.coco       = ExtractCoco(path)
            self.device     = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
            self.prefix     = prefix
            if self.model is not None:
                  self.model.eval()
                  self.model.to(self.device)
            self.boxes_draw = Boxes()
        except ValueError as error:
            print(error)
    
    def running(self,qt_show=10):
        try:
            imgs       = self.coco.extract_dic()
            dic_labels = self.coco.dic_labels
            
            for image in imgs:
                img_p, targets, size = image
                img =  cv2.imread(os.path.join(self.prefix,img_p))

                #preds = inference_detector(self.model, img)
                #box_preds  = (preds['boxes']).clone()
                #box_labels = (preds['labels']).clone()
                #box_scores = (preds['scores']).clone()
                #labels = []
                #scores = []
                #boxes  = []
                #b_gt = []
                #for gt in targets['boxes']:
                #    b_gt.append( [[g.item() for g in gt],1,0] )
                #    gt = (gt.unsqueeze(0)).to(self.device)
                #    distances_dic = {(intersection_over_union(pred.unsqueeze(0),gt)).item():[pred,i] for i,pred in enumerate(box_preds)}
                #    distances     = list(distances_dic.keys())
                #    value         = np.max(distances)

                #    if value >= self.threshold:
                #        box_closed    = distances_dic[value]
                #        labels.append((box_labels[box_closed[1]]).item())
                #        scores.append((box_scores[box_closed[1]]).item())
                #        boxes.append(box_closed[0])

                #        box_scores = [v for i,v in enumerate(box_scores) if box_closed[1] !=i] 
                #        box_labels = [v for i,v in enumerate(box_labels) if box_closed[1] !=i] 
                #        box_preds  = [v for i,v in enumerate(box_preds)  if box_closed[1] !=i] 

                #boxes    = [bx.cpu().tolist() for bx in boxes]
                #boxes_fp = [[bx.cpu().tolist(),box_labels[i].item(),round(box_scores[i].item(),3)] for i,bx in enumerate(box_preds) if box_scores[i].item() >= self.threshold_class]
                #preds_closed ={
                #    'boxes' : torch.tensor(boxes),
                #    'scores': torch.tensor(scores),
                #    'labels': torch.tensor(labels),
                #}
                
                print(targets['labels'])
                if qt_show > 0:
                    tg_boxes = [ [value.cpu().tolist(),0,0] for value in targets['boxes']]
                    img = self.boxes_draw.draw_rectangle(bboxes=tg_boxes, img=img,dim=(640,640),color=(255,0,0))
                    cv2.imwrite('img_'+str(qt_show)+'.png',img)
                    
                qt_show -= 1
               
        except ValueError as error:
            print(error)


if __name__ == '__main__':

    #model     = init_detector('../work_dirs/ssd300_coco/ssd300_coco.py', 'work_dirs/ssd300_coco/best_coco_bbox_mAP_epoch_10.pth', device=torch.device('cuda:0'))
    testing = Testing(
        model=None,
        prefix=os.path.join(os.getcwd(),'../../../dataset/all/train'),
        path='../../../dataset/filesJSON/fold_1_test.json'
        )
    testing.running()