# Traffic Analysis with YOLOv8

Traffic density estimation + event detection (accident/breakdown) + heatmap visualization using YOLOv8 and OpenCV.

This project processes traffic videos and extracts several analytics metrics such as vehicle count, traffic density, average motion-based speed proxy, heatmaps, and accident-like event alerts.

---

## Features

- Vehicle detection using YOLOv8 (car, motorcycle, bus, truck)
- Traffic density estimation (vehicle count + occupancy ratio)
- Speed proxy calculation (frame-to-frame motion)
- Event detection:
  - If a vehicle stops (speed ≈ 0)
  - And density is low
  - For consecutive frames
  - → Marked as **Accident or Breakdown**
- Heatmap generation (temporal vehicle distribution)
- CSV logging for analytics and ML pipelines
- Video rendering with overlays (metrics + alerts + heatmap)

---

## Demo Overview

Pipeline output per frame includes:

- Vehicles detected and classified
- Density label (`Low / Medium / High`)
- Risk score (0–100 heuristic)
- Heatmap overlay
- Event alert text if detected

**Example overlay (conceptual):**

```text
Vehicles: 7
Density: Medium
Risk score: 42
Accident or Breakdown Detected
