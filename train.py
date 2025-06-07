import os
from ultralytics import YOLO
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    model = YOLO('yolov8.yaml').load('yolov8n.pt')
    model.train(data="ultralytics/cfg/datasets/coco128.yaml", 
    Distillation = None, 
    loss_type='None', 
    imgsz=640, 
    epochs=50, 
    batch=32, device=0, workers=0)


if __name__ == '__main__':
    main()

# 916 774 s
# 842 667 n
