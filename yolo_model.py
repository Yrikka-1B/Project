from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model.predict('yrikka-btt-aistudio-2025/BTT_Data/852a64c6-4bd3-495f-8ff7-f5cc85e34316/images/', save =True, save_txt = True, conf=0.25, imgsz = 640, project="preds_dir")  # predict on an image
