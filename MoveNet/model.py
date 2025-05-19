import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from tqdm import tqdm
import time
from PIL import Image

def load_movenet_multipose():
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    return model.signatures['serving_default']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    return input_image, image.shape[1], image.shape[0]  # width, height

def movenet_to_coco_format(output, image_id, width, height, score_threshold=0.2):
    people = output['output_0'].numpy()[0]  # (6, 56)
    results = []

    for person in people:
        person_score = person[55]
        if person_score < score_threshold:
            continue

        ymax, xmax, ymin, xmin = person[51:55]
        x_c = float((xmin+(xmax-xmin)/2) * width) 
        y_c = float((ymin+(ymax-ymin)/2) * height)
        if (width>height):
            h = float((ymax-ymin) * height) * (width/height)
            w = float((xmax-xmin) * width)
        else:
            h = float((ymax-ymin) * height)
            w = float((xmax-xmin) * width) * (height/width)

        keypoints = []
        valid_keypoints = 0
        for i in range(17):
            y, x, kp_score = person[i*3:i*3+3]
            x = float(x * width)
            y = float(y * height)
            v = 2 if kp_score > 0.3 else 1  # visibility
            keypoints.extend([x, y, v])
            if v > 0:
                valid_keypoints += 1

        #y_kp_mean = np.mean([keypoints[i] for i in range(1,len(keypoints),3)]) #+ 30 # shift = 30 from experiments
        #avg = (y_kp_mean + y_c)/2
        if (width>height):
            for i in range(1,len(keypoints),3):
                keypoints[i] = y_c + (keypoints[i]-y_c) * (width/height)
        else:
            for i in range(0,len(keypoints),3):
                keypoints[i] = x_c + (keypoints[i]-x_c) * (height/width)
        #print(y_kp_mean, y_c)
        # Extract bbox [xmin, ymin, xmax, ymax]

        results.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x_c, y_c, w, h],
            "score": float(person_score),
            "keypoints": keypoints
        })
    return results

def main(test_image_dir="../coco/val2017"):
    model = load_movenet_multipose()

    with open("../coco/coco_val_human_images.json", "r") as f:
        image_files = json.load(f)

    results = []
    latency = []

    for img_name in tqdm(image_files):
        img_path = os.path.join(test_image_dir, img_name)
        image_id = int(os.path.splitext(img_name)[0])

        input_image, width, height = preprocess_image(img_path)

        start_t = time.time()
        output = model(input_image)
        end_t = time.time()
        latency.append(end_t - start_t)

        preds = movenet_to_coco_format(output, image_id, width, height)
        results.extend(preds)

    with open("coco_test_predictions.json", "w") as f:
        json.dump(results, f)

    with open("latency.json", "w") as f:
        json.dump(latency, f)

def draw_poses_on_image(image_np, poses, keypoint_thresh=0.3):
    """
    Draws keypoints and bounding boxes for each detected pose on the image.
    Args:
        image_np (np.ndarray): Original image in RGB format.
        poses (List[Dict]): List of poses, each with 'keypoints' (list of 51 floats) and 'score'.
        keypoint_thresh (float): Minimum confidence to draw individual keypoints.
    Returns:
        output_img (np.ndarray): Annotated image in BGR (for OpenCV display/save).
    """
    # Copy and convert to BGR for OpenCV
    output_img = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
    for pose in poses:
        kpts = np.array(pose['keypoints']).reshape(-1, 3)
        # Collect valid keypoints for bbox
        #valid = kpts[kpts[:, 2] >= keypoint_thresh][:, :2]
        #if valid.size > 0:
        #    x_min, y_min = valid.min(axis=0).astype(int)
        #    x_max, y_max = valid.max(axis=0).astype(int)
            # Draw bounding box
        x, y, w, h = [int(i) for i in pose['bbox']]
        #y = y*256/output_img.shape[0]
        x_min, x_max = int(x - w/2), int(x + w/2)
        y_min, y_max = int(y - h/2), int(y + h/2)
        cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Draw keypoints
        for x, y, c in kpts:
            if c < keypoint_thresh:
                continue
            cv2.circle(output_img, (int(x), int(y)), 3, (0, 0, 255), -1)
    return output_img

def minor(test_image_dir="../coco/val2017"):
    model = load_movenet_multipose()

    with open("../coco/coco_val_human_images.json", "r") as f:
        image_files = json.load(f)

    img_path = os.path.join(test_image_dir, image_files[1])
    image_id = int(os.path.splitext(image_files[1])[0])
    orig = cv2.cvtColor(np.array(Image.open(img_path).convert('RGB')), cv2.COLOR_RGB2BGR)

    input_image, width, height = preprocess_image(img_path)
    print(width, height)

    output = model(input_image)

    preds = movenet_to_coco_format(output, image_id, width, height)
    result = draw_poses_on_image(orig, preds, keypoint_thresh=0.3)
    cv2.imwrite('./image.jpg', result)


if __name__ == "__main__":
    main()