from ultralytics import YOLO
import cv2
import fire

def main(path_to_video="../../report/youtube1.mp4", path_to_result="."):
    model = YOLO("yolo11n-pose.pt")
    model.eval()
    results = model.predict(source=path_to_video, show=False, save=True, save_dir=path_to_result)

if __name__ == "__main__":
    fire.Fire(main)
    main()