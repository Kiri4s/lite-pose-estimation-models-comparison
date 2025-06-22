import matplotlib.pyplot as plt
import numpy as np
import json
import cv2

def draw_kpts(id):
    img = cv2.imread("images/train/"+f"{id}.jpg")
    with open("labels/train/"+f"{id}.txt", 'r') as f:
        annotations = f.readlines()

    width = img.shape[1]
    height = img.shape[0]
    for ann in annotations:
        values = [float(i) for i in ann.split()]
        x, y, w, h = values[1:5]
        x = int(x*width)
        y = int(y*height)
        w = int(w*width)
        h = int(h*height)
        #print(x, y, w, h)
        x_min, x_max = x-w//2, x+w//2
        y_min, y_max = y-h//2, y+h//2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        # Draw keypoints
        for i in range(14):
            if values[5+3*i+2] >= 1:
                cv2.circle(img, (int(values[5+3*i]*width), int(values[5+3*i+1]*height)), 1, (0, 0, 255), -1)
    return img

if __name__ == "__main__":
    id = 100000
    fig = draw_kpts(id)
    cv2.imwrite('res.png', fig)
    #img = cv2.imread("images/train/"+f"{id}.jpg")
    #print(img.shape)
