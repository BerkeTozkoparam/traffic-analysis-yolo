import cv2
from ultralytics import YOLO
import csv
import os
import math
import numpy as np

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "yolov8n.pt"
INPUT_VIDEO_PATH = "/Users/berkebarantozkoparan/Desktop/project12/SampleVideo_LowQuality.mp4"
OUTPUT_VIDEO_PATH = "traffic_risk_with_events.mp4"
CSV_OUTPUT_PATH = "traffic_metrics.csv"

# COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

# Confidence threshold
CONF_THRESHOLD = 0.4

# Normalization constants for heuristic risk score
MAX_VEHICLES = 20          # assumed "very heavy traffic"
MAX_OCCUPANCY = 0.4        # 40% of frame area filled by vehicles = very dense
MAX_SPEED_PIXELS = 50.0    # large average movement per frame in pixels

# Event detection parameters
STATIONARY_SPEED_THRESHOLD = 1.0   # pixels per frame, treated as "stopped"
STATIONARY_FRAMES_REQUIRED = 10    # number of consecutive frames to consider it an event


# -------------------------
# Helper functions
# -------------------------
def compute_risk_score(vehicle_count, occupancy_ratio, avg_speed_pixels):
    """
    Compute a heuristic risk score between 0 and 100 based on:
    - vehicle_count
    - occupancy_ratio (0-1)
    - avg_speed_pixels
    This is not a real-world safety model, only a demo metric.
    """
    density_from_count = min(vehicle_count / MAX_VEHICLES, 1.0)
    density_from_occupancy = min(occupancy_ratio / MAX_OCCUPANCY, 1.0)

    density_score = 0.7 * density_from_count + 0.3 * density_from_occupancy
    speed_score = min(avg_speed_pixels / MAX_SPEED_PIXELS, 1.0)

    combined = 0.6 * density_score + 0.4 * speed_score

    return int(combined * 100)


def get_density_label(vehicle_count, occupancy_ratio):
    """
    Simple textual label for traffic density.
    """
    if vehicle_count <= 3 and occupancy_ratio < 0.1:
        return "Low"
    elif vehicle_count <= 8 and occupancy_ratio < 0.25:
        return "Medium"
    else:
        return "High"


# -------------------------
# Load model
# -------------------------
model = YOLO(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")


# -------------------------
# Open input video
# -------------------------
if not os.path.exists(INPUT_VIDEO_PATH):
    print(f"Input video not found: {INPUT_VIDEO_PATH}")
    exit(1)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print("Could not open input video.")
    exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_area = float(width * height)

print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")


# -------------------------
# Prepare output video writer
# -------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)

if not output.isOpened():
    print(f"Could not open output file for writing: {OUTPUT_VIDEO_PATH}")
    cap.release()
    exit(1)


# -------------------------
# Prepare CSV file
# -------------------------
csv_file = open(CSV_OUTPUT_PATH, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_index",
    "vehicle_count",
    "occupancy_ratio",
    "avg_speed_pixels",
    "risk_score",
    "event_flag"
])


# -------------------------
# Heatmap buffer (for visual density)
# -------------------------
# Use float32 for accumulation and decay over time
heatmap_accum = np.zeros((height, width), dtype=np.float32)


# -------------------------
# Frame-by-frame processing
# -------------------------
frame_index = 0
prev_centers = []  # list of (x, y) for previous frame vehicles

# Counter to detect sustained "stopped" situation
stationary_frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO detection (all classes, then we filter by VEHICLE_CLASSES)
    results = model(
        frame,
        stream=False,
        conf=CONF_THRESHOLD
    )

    result = results[0]
    boxes = result.boxes

    vehicle_count = 0
    total_vehicle_area = 0.0
    current_centers = []

    # Temporary mask for heatmap update
    frame_heatmap = np.zeros((height, width), dtype=np.float32)

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in VEHICLE_CLASSES:
                continue

            vehicle_count += 1

            # xyxy: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(max(x1, 0))
            y1 = int(max(y1, 0))
            x2 = int(min(x2, width - 1))
            y2 = int(min(y2, height - 1))

            w = max(x2 - x1, 0)
            h = max(y2 - y1, 0)
            total_vehicle_area += float(w * h)

            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            current_centers.append((cx, cy))

            # Update per-frame heatmap for the area of this vehicle
            if w > 0 and h > 0:
                frame_heatmap[y1:y2, x1:x2] += 1.0

        # Draw all detections on the frame (for visualization)
        frame = result.plot()

    # Compute occupancy ratio
    occupancy_ratio = total_vehicle_area / frame_area if frame_area > 0 else 0.0

    # Estimate average movement (speed proxy) between frames
    avg_speed_pixels = 0.0
    if prev_centers and current_centers:
        distances = []
        for cx, cy in current_centers:
            # Find nearest previous center
            min_dist = None
            for px, py in prev_centers:
                dx = cx - px
                dy = cy - py
                dist = math.sqrt(dx * dx + dy * dy)
                if (min_dist is None) or (dist < min_dist):
                    min_dist = dist
            if min_dist is not None:
                distances.append(min_dist)

        if distances:
            avg_speed_pixels = sum(distances) / len(distances)

    # Update for next frame
    prev_centers = current_centers

    # Compute heuristic risk score
    risk_score = compute_risk_score(vehicle_count, occupancy_ratio, avg_speed_pixels)
    density_label = get_density_label(vehicle_count, occupancy_ratio)

    # -------------------------
    # Event detection:
    # "Accident or Breakdown" if average speed is near zero
    # and traffic density is not high, for several consecutive frames
    # -------------------------
    event_flag = 0
    if vehicle_count > 0 and density_label == "Low" and avg_speed_pixels < STATIONARY_SPEED_THRESHOLD:
        stationary_frame_counter += 1
    else:
        stationary_frame_counter = 0

    if stationary_frame_counter >= STATIONARY_FRAMES_REQUIRED:
        event_flag = 1

    # -------------------------
    # Update global heatmap
    # -------------------------
    # Decay previous values to keep the heatmap temporal
    heatmap_accum *= 0.95
    # Add current frame mask
    heatmap_accum += frame_heatmap

    # Normalize heatmap to [0, 255]
    heatmap_norm = heatmap_accum.copy()
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Apply color map to create heatmap image
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend heatmap with original frame
    alpha = 0.4  # heatmap opacity
    frame = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)

    # -------------------------
    # Draw metrics on frame
    # -------------------------
    text_1 = f"Vehicles: {vehicle_count}"
    text_2 = f"Density: {density_label}"
    text_3 = f"Risk score: {risk_score}"

    cv2.putText(
        frame,
        text_1,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text_2,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text_3,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # If an accident/breakdown-like situation is detected, show alert
    if event_flag == 1:
        alert_text = "Accident or Breakdown Detected"
        cv2.putText(
            frame,
            alert_text,
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    # Save frame to output video
    output.write(frame)

    # Log to CSV
    csv_writer.writerow([
        frame_index,
        vehicle_count,
        round(occupancy_ratio, 4),
        round(avg_speed_pixels, 2),
        risk_score,
        event_flag
    ])

    # Optionally display
    cv2.imshow("Traffic density, risk and events", frame)

    frame_index += 1
    if frame_index % 50 == 0:
        print(f"Processed {frame_index} / {frame_count} frames")

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------
cap.release()
output.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Done. Output video: {OUTPUT_VIDEO_PATH}")
print(f"Metrics saved to CSV: {CSV_OUTPUT_PATH}")
