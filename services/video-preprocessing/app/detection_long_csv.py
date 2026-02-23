from __future__ import annotations

import csv
import os
from typing import Dict, List

import cv2
from ultralytics import YOLO


DEFAULT_MODEL_PATH = os.getenv("LONG_VIDEO_YOLO_MODEL", "yolov8l.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("LONG_VIDEO_YOLO_CONF", "0.3"))
IOU_THRESHOLD = float(os.getenv("LONG_VIDEO_YOLO_IOU", "0.3"))

PERSON_CLASS = 0
COW_CLASS = 19


def load_model(model_path: str | None = None) -> YOLO:
    path = model_path or DEFAULT_MODEL_PATH
    return YOLO(path)


def detect_frame(model: YOLO, frame) -> Dict[str, List[Dict]]:
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

    detection_result = {"humans": [], "cows": [], "total_detections": 0}
    if results[0].boxes is None:
        return detection_result

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    detection_result["total_detections"] = len(boxes)

    for box, cls, conf in zip(boxes, classes, confidences):
        class_id = int(cls)
        if class_id == PERSON_CLASS:
            detection_result["humans"].append(
                {"box": box.astype(int).tolist(), "confidence": float(conf)}
            )
        elif class_id == COW_CLASS:
            detection_result["cows"].append(
                {"box": box.astype(int).tolist(), "confidence": float(conf)}
            )

    return detection_result


def process_video_to_csv(video_path: str, output_csv_path: str, model: YOLO | None = None) -> bool:
    print(f"Processing long video: {os.path.basename(video_path)}")

    if model is None:
        model = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps}fps, {total_frames} frames")

    csv_data: List[List] = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_frame(model, frame)

        for human in detections["humans"]:
            x1, y1, x2, y2 = human["box"]
            conf = human["confidence"]
            csv_data.append([frame_count, "person", conf, x1, y1, x2, y2])

        for cow in detections["cows"]:
            x1, y1, x2, y2 = cow["box"]
            conf = cow["confidence"]
            csv_data.append([frame_count, "cow", conf, x1, y1, x2, y2])

        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames else 0
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

    cap.release()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_idx", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(csv_data)

    print(f"Processing complete: {frame_count} frames, {len(csv_data)} detections")
    print(f"CSV saved to: {output_csv_path}")
    return True
