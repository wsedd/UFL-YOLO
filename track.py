import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('') # select your model.pt path
    model.track(source='video.mp4',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )