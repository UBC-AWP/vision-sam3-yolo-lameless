import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


def is_box_fully_visible(box):
    x1, y1, x2, y2 = box
    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def calculate_speed(prev_box, curr_box):
    if prev_box is None or curr_box is None:
        return 0

    prev_center_x = (prev_box[0] + prev_box[2]) / 2
    prev_center_y = (prev_box[1] + prev_box[3]) / 2
    curr_center_x = (curr_box[0] + curr_box[2]) / 2
    curr_center_y = (curr_box[1] + curr_box[3]) / 2

    distance = np.sqrt((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2)
    return distance


def calculate_entropy(values):
    if len(values) == 0:
        return 0.0
    values = np.array(values)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    probs = values / values.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def simple_track_detections(df, max_distance=100):
    df = df.copy()
    df["track_id"] = -1

    df = df.sort_values("frame_idx")

    current_tracks = {}
    next_track_id = 0

    for frame_idx in df["frame_idx"].unique():
        frame_detections = df[df["frame_idx"] == frame_idx]

        for idx, detection in frame_detections.iterrows():
            box = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            best_track_id = None
            min_distance = max_distance

            for track_id, track_info in current_tracks.items():
                if track_info["last_frame"] < frame_idx - 5:
                    continue

                dist = np.sqrt((center_x - track_info["center_x"]) ** 2 + (center_y - track_info["center_y"]) ** 2)
                if dist < min_distance:
                    min_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                df.at[idx, "track_id"] = best_track_id
                current_tracks[best_track_id]["last_frame"] = frame_idx
                current_tracks[best_track_id]["center_x"] = center_x
                current_tracks[best_track_id]["center_y"] = center_y
            else:
                df.at[idx, "track_id"] = next_track_id
                current_tracks[next_track_id] = {
                    "last_frame": frame_idx,
                    "center_x": center_x,
                    "center_y": center_y,
                }
                next_track_id += 1

    return df


def smooth_coordinates(df, window: int, method: str = "median"):
    if window is None or window <= 1:
        return df
    df = df.sort_values("frame_idx").copy()
    cols = ["x1", "y1", "x2", "y2"]
    roll = df[cols].rolling(window=window, center=True, min_periods=1)
    if method == "mean":
        smoothed = roll.mean()
    else:
        smoothed = roll.median()
    df[cols] = smoothed
    return df


def analyze_video_features(csv_path, segment_size=30, smooth_window=5, smooth_method="median"):
    try:
        df = pd.read_csv(csv_path)
        cow_detections = df[df["class"].str.contains("cow", case=False, na=False)]

        if len(cow_detections) < 30:
            return {
                "insufficient_detections": True,
                "detection_count": len(cow_detections),
            }

        cow_detections = cow_detections.sort_values("frame_idx")
        cow_detections = smooth_coordinates(cow_detections, window=smooth_window, method=smooth_method)

        aspect_ratios = []
        top_right_x = []
        top_right_y = []
        speeds = []
        areas = []
        prev_box = None

        for _, detection in cow_detections.iterrows():
            box = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
            if is_box_fully_visible(box):
                width = box[2] - box[0]
                height = box[3] - box[1]
                if height > 0:
                    aspect_ratio = width / height
                    aspect_ratios.append(aspect_ratio)

                area = width * height
                areas.append(area)

                top_right_x.append(box[2])
                top_right_y.append(box[3])

                if prev_box is not None:
                    speed = calculate_speed(prev_box, box)
                    speeds.append(speed)
                prev_box = box

        if len(aspect_ratios) < 30 or len(top_right_x) < 30:
            return {
                "insufficient_detections": True,
                "detection_count": len(cow_detections),
                "valid_detections": len(aspect_ratios),
            }

        features: Dict[str, float] = {}

        features["mean_ratio"] = float(np.mean(aspect_ratios))
        std_ratio = np.std(aspect_ratios)
        features["ratio_stability"] = float(1.0 / (1.0 + std_ratio))

        if len(top_right_x) > 1:
            x_values = np.array(top_right_x).reshape(-1, 1)
            y_values = np.arange(len(top_right_x))
            lr = LinearRegression()
            lr.fit(x_values, y_values)
            features["x_linearity"] = float(lr.score(x_values, y_values))
        else:
            features["x_linearity"] = 0.0

        features["y_stability"] = float(1.0 / (1.0 + np.std(top_right_y)))

        if len(top_right_x) > 1:
            increasing_pairs = sum(1 for i in range(1, len(top_right_x)) if top_right_x[i] > top_right_x[i - 1])
            features["monotonic_score"] = float(increasing_pairs / (len(top_right_x) - 1))
        else:
            features["monotonic_score"] = 0.0

        segments = []
        current_segment = []
        for i, ratio in enumerate(aspect_ratios):
            current_segment.append(ratio)
            if len(current_segment) >= segment_size or i == len(aspect_ratios) - 1:
                if len(current_segment) >= 10:
                    segments.append(current_segment.copy())
                current_segment = []

        if segments:
            segment_min_ratios = [min(segment) for segment in segments]
            features["min_segment_ratio"] = float(min(segment_min_ratios))
        else:
            features["min_segment_ratio"] = float(features["mean_ratio"])

        if len(speeds) > 1:
            features["speed_cv_mean"] = float(np.std(speeds) / np.mean(speeds)) if np.mean(speeds) > 0 else 0.0
        else:
            features["speed_cv_mean"] = 0.0

        if len(areas) > 0:
            features["mean_area"] = float(np.mean(areas))
            std_area = np.std(areas)
            features["area_stability"] = float(1.0 / (1.0 + std_area / (features["mean_area"] + 1e-6)))
        else:
            features["mean_area"] = 0.0
            features["area_stability"] = 0.0

        if len(aspect_ratios) >= 20 and len(areas) >= 20:
            center_start = int(len(aspect_ratios) * 0.3)
            center_end = int(len(aspect_ratios) * 0.7)
            center_ratios = aspect_ratios[center_start:center_end]
            center_areas = areas[center_start:center_end]

            if len(center_ratios) > 0:
                center_std_ratio = np.std(center_ratios)
                features["center_ratio_stability"] = float(1.0 / (1.0 + center_std_ratio))
            else:
                features["center_ratio_stability"] = float(features["ratio_stability"])

            if len(center_areas) > 0:
                center_mean_area = np.mean(center_areas)
                center_std_area = np.std(center_areas)
                features["center_area_stability"] = float(1.0 / (1.0 + center_std_area / (center_mean_area + 1e-6)))
            else:
                features["center_area_stability"] = float(features["area_stability"])
        else:
            features["center_ratio_stability"] = float(features["ratio_stability"])
            features["center_area_stability"] = float(features["area_stability"])

        frame_data = {}
        for _, detection in cow_detections.iterrows():
            frame_idx = detection["frame_idx"]
            box = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
            if is_box_fully_visible(box):
                if frame_idx not in frame_data:
                    frame_data[frame_idx] = []
                frame_data[frame_idx].append(box)

        if len(frame_data) < 10:
            for key in [
                "large_box_concurrency_mean",
                "vertical_band_entropy",
                "stop_go_count",
                "stop_go_ratio",
                "track_fragmentation_mean",
                "heading_consistency",
            ]:
                features[key] = 0.0
        else:
            screen_area = 1920 * 1080
            area_threshold = screen_area * 0.1

            large_box_counts = []
            for frame_idx, boxes in frame_data.items():
                count = 0
                for box in boxes:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area >= area_threshold:
                        count += 1
                large_box_counts.append(count)

            if len(large_box_counts) > 0:
                features["large_box_concurrency_mean"] = float(np.mean(large_box_counts))
            else:
                features["large_box_concurrency_mean"] = 0.0

            K = 5
            band_height = 1080 / K
            band_counts = {i: [] for i in range(K)}

            for frame_idx, boxes in frame_data.items():
                frame_band_counts = {i: 0 for i in range(K)}
                for box in boxes:
                    center_y = (box[1] + box[3]) / 2
                    band_idx = min(int(center_y / band_height), K - 1)
                    frame_band_counts[band_idx] += 1

                for i in range(K):
                    band_counts[i].append(frame_band_counts[i])

            band_activities = [np.mean(band_counts[i]) if len(band_counts[i]) > 0 else 0 for i in range(K)]
            features["vertical_band_entropy"] = float(calculate_entropy(band_activities))

            speed_threshold = 5.0
            stop_frames = []
            for i in range(len(speeds)):
                if speeds[i] < speed_threshold:
                    stop_frames.append(1)
                else:
                    stop_frames.append(0)

            if len(stop_frames) > 0:
                stop_segments = 0
                in_stop = False
                for val in stop_frames:
                    if val == 1 and not in_stop:
                        stop_segments += 1
                        in_stop = True
                    elif val == 0:
                        in_stop = False

                features["stop_go_count"] = float(stop_segments)
                features["stop_go_ratio"] = float(sum(stop_frames) / len(stop_frames))
            else:
                features["stop_go_count"] = 0.0
                features["stop_go_ratio"] = 0.0

            try:
                tracked_df = simple_track_detections(cow_detections, max_distance=150)

                track_segments = {}
                for track_id in tracked_df["track_id"].unique():
                    if track_id < 0:
                        continue
                    track_frames = tracked_df[tracked_df["track_id"] == track_id]["frame_idx"].values
                    if len(track_frames) > 0:
                        segments = 1
                        for i in range(1, len(track_frames)):
                            if track_frames[i] - track_frames[i - 1] > 1:
                                segments += 1
                        duration = len(track_frames)
                        if duration > 0:
                            track_segments[track_id] = segments / duration

                if len(track_segments) > 0:
                    features["track_fragmentation_mean"] = float(np.mean(list(track_segments.values())))
                else:
                    features["track_fragmentation_mean"] = 0.0

                headings = []
                for track_id in tracked_df["track_id"].unique():
                    if track_id < 0:
                        continue
                    track_data = tracked_df[tracked_df["track_id"] == track_id].sort_values("frame_idx")
                    if len(track_data) > 1:
                        for i in range(1, len(track_data)):
                            prev_row = track_data.iloc[i - 1]
                            curr_row = track_data.iloc[i]
                            dx = (curr_row["x1"] + curr_row["x2"]) / 2 - (prev_row["x1"] + prev_row["x2"]) / 2
                            dy = (curr_row["y1"] + curr_row["y2"]) / 2 - (prev_row["y1"] + prev_row["y2"]) / 2
                            if dx != 0 or dy != 0:
                                angle = np.arctan2(dy, dx)
                                headings.append(angle)

                if len(headings) > 1:
                    headings = np.array(headings)
                    cos_mean = np.mean(np.cos(headings))
                    sin_mean = np.mean(np.sin(headings))
                    circular_variance = 1 - np.sqrt(cos_mean**2 + sin_mean**2)
                    features["heading_consistency"] = float(1 - circular_variance)
                else:
                    features["heading_consistency"] = 0.0

            except Exception:
                features["track_fragmentation_mean"] = 0.0
                features["heading_consistency"] = 0.0

        frame_width = float(cow_detections["x2"].max()) if len(cow_detections) > 0 else 1920.0
        frame_height = float(cow_detections["y2"].max()) if len(cow_detections) > 0 else 1080.0

        if len(speeds) > 0:
            features["mean_speed_raw"] = float(np.mean(speeds))
        else:
            features["mean_speed_raw"] = 0.0

        if len(speeds) > 1:
            features["sd_speed_raw"] = float(np.std(speeds))
        else:
            features["sd_speed_raw"] = 0.0

        if len(areas) > 1:
            features["sd_area_raw"] = float(np.std(areas))
        else:
            features["sd_area_raw"] = 0.0

        if len(aspect_ratios) > 1:
            features["sd_aspect_raw"] = float(np.std(aspect_ratios))
        else:
            features["sd_aspect_raw"] = 0.0

        total_frames = len(frame_data) if len(frame_data) > 0 else 1
        touch_horizontal_frames = 0
        touch_vertical_frames = 0

        for frame_idx, boxes in frame_data.items():
            frame_touch_left = False
            frame_touch_right = False
            frame_touch_top = False
            frame_touch_bottom = False

            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= 5:
                    frame_touch_left = True
                if (frame_width - x2) <= 5:
                    frame_touch_right = True
                if y1 <= 5:
                    frame_touch_top = True
                if (frame_height - y2) <= 5:
                    frame_touch_bottom = True

            if frame_touch_left or frame_touch_right:
                touch_horizontal_frames += 1
            if frame_touch_top or frame_touch_bottom:
                touch_vertical_frames += 1

        features["touch_ratio_horizontal"] = float(touch_horizontal_frames / total_frames) if total_frames > 0 else 0.0
        features["touch_ratio_vertical"] = float(touch_vertical_frames / total_frames) if total_frames > 0 else 0.0

        features["file_name"] = os.path.basename(csv_path)
        features["insufficient_detections"] = False
        return features

    except Exception as e:
        print(f"Failed to analyze file: {csv_path}, error: {e}")
        return None
