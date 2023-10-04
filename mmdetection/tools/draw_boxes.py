from pycocotools.coco import COCO
import cv2
import numpy as np
import os

# This is my references: https://gist.github.com/interactivetech/c2913317603b79c02ff49fa9824f1104
class Boxes(object):

    def draw_text(self, img, text, position, color=(0,255,0)):
        try:
            img = cv2.putText(img, text,position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color , 1, cv2.LINE_AA)
            return img
        except ValueError as error:
            print(error)

    def draw_rectangle(self, bboxes, img, color=(0,255,0), dim=(200,200), classes=None):
        try:
            for bbox in bboxes:
                prob = None
                if len(bbox) > 1:
                    if len(bbox) > 2:
                        prob = bbox[2]
                    index = bbox[1]
                    bbox  = bbox[0]

                bbox = [int(x) for x in bbox]
                point_1 = (bbox[0], bbox[1])
                point_2 = (bbox[2]+bbox[0] if bbox[2]+bbox[0]< dim[0] else dim[0]-2, bbox[3]+bbox[1] if bbox[3]+bbox[1] < dim[1] else dim[1]-2)
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
            
    def fromJson(self,path, images, labels=('pig',)):
        try:
            coco    = COCO(path)
            classes = {cat['id']:labels.index(cat['name']) for cat in coco.cats.values()}
            ground  = {}
            for key in coco.imgs.keys():
                anns         = coco.loadAnns(coco.getAnnIds(imgIds=key))
                height,width = int(coco.imgs[key]['height']),int(coco.imgs[key]['width'])
                bboxes       = [[annotation['bbox'], classes[annotation['category_id']]] for annotation in anns]
                ground[ coco.imgs[key]['file_name'] ] = {'bboxes': bboxes,'height':height,'width':width}
            self.metrics(images ,ground ,ground.copy())
            return ground
        except ValueError as error:
            print(error)

    def metrics(self, path, annots, preds, threshold=0.8, labels=('pig',), number_show=5):
        try:
            imgs = self.more_close(annots.copy(),preds.copy(),threshold)
            tp = fp = fn = 0
            for key, values in imgs.items():

                metric ={'tp':0,'fp':0, 'fn':0}
                if number_show > 0:
                    name_img = os.path.join(path, key)
                    dim      = values['dim']
                    img      = cv2.resize( cv2.imread( name_img ) , dim , interpolation = cv2.INTER_AREA)
                    img = self.draw_rectangle(values['tp'] ,img,dim=dim,color=(0,255,0),classes=labels)
                    img = self.draw_rectangle(values['fp'] ,img,dim=dim,color=(255,255,0), classes=labels)
                    img = self.draw_rectangle(values['fn'] ,img,dim=dim,color=(0,0,255), classes=labels)
                    img = self.draw_rectangle( annots[key]['bboxes'], img, dim=dim,color=(255,0,0), classes=labels)
                number_show -= 1
                
                count = 15
                for key in metric.keys():
                    metric[key] += len(values[key])
                    img = self.draw_text(img,key+':'+str(metric[key]),(1,count),(255,0,0))
                    count+=20 
                
                tp += metric['tp']
                fp += metric['fp']
                fn += metric['fn']

                if number_show > 0:
                    cv2.imshow('Saida', img)
                    cv2.waitKey(0)

            print('Results final..: ', tp, fn, fp)
            precision = self.operation(numerator=tp,denominator=(tp+fp))
            recall    = self.operation(numerator=tp, denominator=(tp+fn))
        except ValueError as error:
            print(error)

    def save_file(self,data):
        try:
            with open('results.csv', 'a') as file:
                for d in data:
                    file.write(d,',')
                file.close()
        except ValueError as error:
            print(error)
            
    def operation(self, numerator, denominator, precision=3):
        try:
            value = round(numerator/denominator, precision)
            return value
        except ZeroDivisionError:
            return 0.000
        
    def more_close(self, annots, pred, threshold=0.8):
        try:
            imgs = {}
            for key, value in annots.items():
                tp = []
                fn = []
                if key in pred:
                   box_ann  = value['bboxes']
                   box_pred = pred[key]['bboxes'].copy()
                   for b_a in box_ann:
                       distance = [self.iou(b_a[0],b_p[0]) for b_p in box_pred]
                       position = np.argmax(distance)
                       if distance[position] >= threshold:
                           box = box_pred[position]
                           if box[1] == b_a[1]:
                               tp.append(box.copy())
                           else:
                               fn.append(box.copy())
                           del box_pred[position]
                imgs[key] = {'tp':tp, 'fn':fn,'fp':box_pred,'dim':(value['width'],value['height'])}
            return imgs
        except ValueError as error:
            print(error)
  
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

if __name__ == '__main__':
    draw = Boxes()
    gt = draw.fromJson('../../../detectores_json_k_dobras/dataset/filesJSON/fold_10_test.json','../../../detectores_json_k_dobras/dataset/all/train/')