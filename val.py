import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/UFL-YOLO/weights/best.pt')  
    model.val(data='dataset/URPC2020/data.yaml',
              split='val',  
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='UFL-YOLO',
              )
