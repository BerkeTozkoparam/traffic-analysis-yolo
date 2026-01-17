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

## Technology Stack

- Python 3.10+
- Ultralytics YOLOv8
- OpenCV
- NumPy

---

## Requirements (pip packages)

These dependencies are required:

```txt
ultralytics
opencv-python
numpy
If you use GPU or Apple Silicon, PyTorch / Metal backend can be added optionally.
Installation & Setup
1. Clone the repository
git clone https://github.com/BerkeTozkoparam/traffic-analysis-yolo.git
cd traffic-analysis-yolo
2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install ultralytics opencv-python numpy
Usage
Place your input video file in the project directory.
Run the script:
python main.py
After processing, outputs will include:
traffic_risk_with_events.mp4
traffic_metrics.csv
Output Files
Video Output
Rendered video with:
Bounding boxes
Traffic metrics
Event alerts
Heatmap overlay
CSV Output
Structured values for analytics:
frame_index	vehicle_count	occupancy_ratio	avg_speed_pixels	risk_score	event_flag
132	5	0.11	3.2	37	0
133	5	0.11	0.4	35	1
event_flag = 1 indicates detected accident/breakdown.
Configuration
All configuration parameters are at the top of main.py:
CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]
STATIONARY_SPEED_THRESHOLD = 1.0
STATIONARY_FRAMES_REQUIRED = 10
These control detection, vehicle filtering, and event sensitivity.
Detection Logic Summary
Traffic Density
Traffic density is estimated from:
vehicle count
occupancy ratio (vehicle_area / frame_area)
Density label → Low / Medium / High
Event (Accident/Breakdown)
Event is detected when:
avg_speed ≈ 0
density == Low
sustained N frames (configurable)
Heatmap
Heatmap shows vehicle distribution over time using:
accumulation matrix
temporal decay
cv2.applyColorMap visualization
blended overlay (cv2.addWeighted)
Roadmap / Future Enhancements
Planned improvements:
Lane-specific vehicle counting
Real speed estimation with homography calibration
Multi-camera fusion for highways
Tracking-based vehicle IDs (DeepSORT / ByteTrack)
Live dashboard alerts (Flask/FastAPI/Streamlit)
ROS integration for ITS systems
ONNX / TensorRT acceleration
Web-stream support (RTSP/RTMP)
Screenshots / Demo (Optional)
If you add images or GIFs, insert here:
./demo/frame001.jpg
./demo/output.gif
Author
Berke Baran Tozkoparan
AI / Computer Vision / Industrial Engineering projects

