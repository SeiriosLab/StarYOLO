from ultralytics import YOLO
model = YOLO('StarBL-YOLO.yaml')
results = model.train(data='cherry-tomato-pose.yaml', epochs=1000,lr0=0.001, batch=16, save_dir='./runs/detect/train/Star-BL-YOLO',channel=4,device=0)
