import os
from ultralytics import YOLO
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    model_t = YOLO('runs/detect/train_v8s/weights/best.pt')  # the teacher model
    model_s = YOLO('runs/detect/train_v8n/weights/best.pt')  # the student model
    """
    Attributes:
        Distillation: the distillation model
        loss_type: mgd, cwd
        amp: Automatic Mixed Precision
    """
    model_s.train(data="ultralytics/cfg/datasets/coco128.yaml", 
    Distillation=model_t.model, 
    loss_type='mgd', 
    amp=False, 
    imgsz=640, 
    epochs=100,
    batch=32, device=0, workers=0, lr0=0.001)


if __name__ == '__main__':
    main()
