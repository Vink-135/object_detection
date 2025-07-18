# yolo_utils.py
from ultralytics import YOLO
import cv2
from collections import Counter
import os

model = YOLO("yolov8n.pt")  # Load once globally

def process_video(video_path, output_path="processed_output.mp4"):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25  # fallback

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4, verbose=False)
        result = results[0]

        labels = result.names
        detected_classes = result.boxes.cls.tolist()
        detected_names = [labels[int(cls)] for cls in detected_classes]
        counts = Counter(detected_names)

        count_text = "  ".join([f"{cls}: {cnt}" for cls, cnt in counts.items()])
        annotated_frame = result.plot()

        cv2.rectangle(annotated_frame, (10, 10), (frame_width - 10, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, count_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path