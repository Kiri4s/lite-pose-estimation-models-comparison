from ultralytics import YOLO
import os
import json
from tqdm import tqdm

model = YOLO("yolov8n.pt") 

test_image_dir = "./val2017/"
image_paths = [os.path.join(test_image_dir, img) for img in os.listdir(test_image_dir)]

human_test_images = []

for img_path in tqdm(image_paths):
    results = model(img_path, classes=[0], verbose=False)  # Class 0 = "person" in COCO
    if len(results[0].boxes) > 0:  # If humans are detected
        filename = os.path.basename(img_path)
        human_test_images.append(filename)


with open("coco_val_human_images.json", "w") as f:
    json.dump(human_test_images, f)