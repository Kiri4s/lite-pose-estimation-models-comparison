import json
from pycocotools.coco import COCO
import numpy as np

def compute_oks(gt_kp, dt_kp, area):
    """Compute Object Keypoint Similarity (OKS) between ground truth and detected keypoints."""
    # Define sigmas for COCO person keypoints (from pycocotools)
    sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    gt_kp = np.array(gt_kp).reshape(-1, 3)
    dt_kp = np.array(dt_kp).reshape(-1, 3)
    gt_x, gt_y, gt_v = gt_kp[:, 0], gt_kp[:, 1], gt_kp[:, 2]
    dt_x, dt_y = dt_kp[:, 0], dt_kp[:, 1]
    visible = gt_v > 0
    if not np.any(visible):
        return 0
    d2 = (gt_x[visible] - dt_x[visible])**2 + (gt_y[visible] - dt_y[visible])**2
    var = 8 * np.array(sigmas)[visible]**2 * area
    oks_per_kp = np.exp(-d2 / var)
    return np.mean(oks_per_kp)

def pck_accuracy_4_every_keypoint(gt, dt, save2file = ''):
    img_ids = gt.getImgIds()

    num_kp = 17
    correct_pck = np.zeros(num_kp)
    correct_acc = np.zeros(num_kp)
    total_visible = np.zeros(num_kp)

    # Threshold parameters
    alpha = 0.1  # PCK threshold factor
    fixed_threshold = 10  # Accuracy threshold in pixels
    oks_threshold = 0.5  # OKS threshold for matching

    for img_id in img_ids:
        gt_ann_ids = gt.getAnnIds(imgIds=img_id)
        gt_anns = gt.loadAnns(gt_ann_ids)
        dt_ann_ids = dt.getAnnIds(imgIds=img_id)
        dt_anns = dt.loadAnns(dt_ann_ids)
        
        if not gt_anns or not dt_anns:
            continue
        
        # Compute OKS matrix
        oks_matrix = np.zeros((len(dt_anns), len(gt_anns)))
        for d, dt_ann in enumerate(dt_anns):
            for g, gt_ann in enumerate(gt_anns):
                oks_matrix[d, g] = compute_oks(gt_ann['keypoints'], dt_ann['keypoints'], gt_ann['area'])
        
        # Match detections to ground truths (greedy matching)
        matches = []
        used_dt = set()
        for g in range(len(gt_anns)):
            if not any(d not in used_dt for d in range(len(dt_anns))):
                break
            d_scores = [(d, oks_matrix[d, g]) for d in range(len(dt_anns)) if d not in used_dt]
            if d_scores:
                d, oks = max(d_scores, key=lambda x: x[1])
                if oks > oks_threshold:
                    matches.append((g, d))
                    used_dt.add(d)
        
        # Evaluate keypoints for matched pairs
        for g, d in matches:
            gt_ann = gt_anns[g]
            dt_ann = dt_anns[d]
            gt_kp = np.array(gt_ann['keypoints']).reshape(-1, 3)
            dt_kp = np.array(dt_ann['keypoints']).reshape(-1, 3)
            scale = np.sqrt(gt_ann['area'])
            
            for i in range(num_kp):
                if gt_kp[i, 2] == 2:  # Visible keypoint
                    dist = np.sqrt((gt_kp[i, 0] - dt_kp[i, 0])**2 + (gt_kp[i, 1] - dt_kp[i, 1])**2)
                    if dist < alpha * scale:
                        correct_pck[i] += 1
                    if dist < fixed_threshold:
                        correct_acc[i] += 1
                    total_visible[i] += 1

    # Compute PCK and Accuracy per keypoint type
    pck = [correct_pck[i] / total_visible[i] if total_visible[i] > 0 else 0 for i in range(num_kp)]
    accuracy = [correct_acc[i] / total_visible[i] if total_visible[i] > 0 else 0 for i in range(num_kp)]

    # Define keypoint names
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    if save2file != '':
        with open(save2file, 'w') as f:
            f.write("PCK@0.1\n")
            for i, name in enumerate(keypoint_names):
                f.write(f"{pck[i]:.4f}\n")
            f.write("\nAccuracy\n")
            for i, name in enumerate(keypoint_names):
                f.write(f"{accuracy[i]:.4f}\n")

    print("Keypoint Metrics:")
    print("-----------------")
    for i, name in enumerate(keypoint_names):
        print(f"{name}:")
        print(f"  PCK@0.1 = {pck[i]:.4f}")
        print(f"  Accuracy (10px) = {accuracy[i]:.4f}")

if __name__ == "__main__":
    gt_file = 'reference_keypoints.json'
    dt_file = 'detected_keypoints.json'
    gt = COCO(gt_file)
    dt = gt.loadRes(dt_file)
    pck_accuracy_4_every_keypoint(gt, dt)