from ultralytics import YOLO
model = YOLO('.../best.pt')
results=model(task='pose', source ="your_source_file", save=True,iou=0.5,conf=0.25,save_dir='./runs/detect/test/Star-BL-YOLO',channel=4,save_txt=True,device=0)
