from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    model.eval()