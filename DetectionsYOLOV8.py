import sys
import cv2
import numpy as np
import os
import time
import supervision
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
from ultralytics import YOLO
from tkinter import filedialog as fd 



class resultYOLO:
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))
    def result(frame,modelName):
        MODEL=modelName
        model = YOLO(MODEL)
        model.fuse()
        CLASS_NAMES_DICT = model.model.names
        CLASS_ID = [0]
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=0, text_thickness=0, text_scale=0)
        results = model(frame)
        #print(byte_tracker)
        detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
        labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
        #frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        #cv2.imshow("Nome", frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        lista = []
        classes_usadas = []
        classes = [0]
        for j in classes:
            lista1 = []
            for i in range(len(detections.xyxy)):
                if detections.class_id[i] == j:
                    classes_usadas.append(j)
                    lista1.append([detections.xyxy[i][0],detections.xyxy[i][1],detections.xyxy[i][2],detections.xyxy[i][3],detections.confidence[i]])
            lista.append(np.array(lista1,dtype='float32'))

        return lista