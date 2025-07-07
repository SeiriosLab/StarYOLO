from ultralytics import YOLO
model = YOLO('.../your.pt_file')
validation_results = model.val(data='cherry-tomato-pose-val.yaml',
                               imgsz=640,
                               batch=1,
                               conf=0.001,
                               iou=0.6,
                               device='0',
                               channel=4,
                               save_dir= './runs/detect/val/Star-BL-YOLO')
