from ultralytics import YOLO
import sys
import gc
def TreinoYOLOV8():
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    model.train(
            data='data.yaml',
                epochs=3, 
                imgsz=640, 
                patience = 50,
                batch = 64,
                project = 'YOLOV8',
                #name = 'Contador_De_Alevinos',
                exist_ok = True,
                optimizer = 'AdamW',
                single_cls = True,
                rect = True,
                cos_lr = True,
                lr0 = 0.001, #Taxa De Aprendizado Inicial
                lrf = 0.001,#Taxa de Aprendizado Final
                val = False,
                plots = False
    )	