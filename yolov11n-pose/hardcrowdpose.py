from ultralytics import YOLO
import cv2
import json
import os
import time
from tqdm import tqdm

def main(test_image_dir="../CrowdPose/dataset/images/test"):
    model = YOLO("weights/yolo11n-pose.pt")
    model.eval()

    with open("../CrowdPose/test_file_names.json", "r") as f:
        image_files = json.load(f)

    results = []
    latency = []
    for img_name in tqdm(image_files):
        img_path = os.path.join(test_image_dir, img_name)

        start_t = time.time()
        predictions = model(img_path)[0]
        end_t = time.time()
        latency.append(end_t-start_t)

        image_id = int(os.path.splitext(img_name)[0])

        boxes = predictions.boxes.xywh.cpu().numpy()  # BBox in [x_center, y_center, width, height]
        scores = predictions.boxes.conf.cpu().numpy()
        keypoints = predictions.keypoints.xy.cpu().numpy()

        for box, score, kps in zip(boxes, scores, keypoints):
            # keypoints to COCO format: [x1, y1, v1, x2, y2, v2, ...] (17 keypoints)
            kp_list = []
            for x, y in kps:
                kp_list.extend([float(x), float(y), 2])  # COCO expects 17 keypoints in order
            kp_list_crowdpose = []
            for i in range(5,17):
                kp_list_crowdpose.extend(kp_list[3*i:3*i+3])
            kp_list_crowdpose.extend(kp_list[0:3])
            kp_list_crowdpose.extend([(kp_list[15]+kp_list[18])/2, (kp_list[16]+kp_list[19])/2, 2])
            
            results.append({
                "image_id": image_id,
                "category_id": 1,  # COCO "person" class ID
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score),
                "keypoints": kp_list_crowdpose
            })

    with open("results/hardcrowdpose_test_predictions.json", "w") as f:
        json.dump(results, f)

    with open("results/hardlatency_crowdpose.json", 'w') as f:
        json.dump(latency, f)

if __name__ == "__main__":
    main()