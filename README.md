Traffic Analysis with YOLOv8
Traffic density estimation + event detection (accident/breakdown) + heatmap visualization using YOLOv8 and OpenCV.
This project processes traffic videos and extracts several analytics metrics such as vehicle count, traffic density, average motion-based speed proxy, heatmaps, and accident-like event alerts.
Features
✔ Vehicle detection using YOLOv8 (car, motorcycle, bus, truck)
✔ Traffic density estimation (vehicle count + occupancy ratio)
✔ Speed proxy calculation (frame-to-frame motion)
✔ Event detection:
If a vehicle stops (speed ≈ 0)
And density is low
For consecutive frames
→ Marked as Accident or Breakdown
✔ Heatmap generation (temporal vehicle distribution)
✔ CSV logging for analytics and ML pipelines
✔ Video rendering with overlays (metrics + alerts + heatmap)
Demo Overview
Pipeline output per frame:
Vehicles detected and classified
Density label: Low / Medium / High
Risk score (0–100 heuristic)
Heatmap overlay
Event alert text if detected
Example overlay (conceptual):
Vehicles: 7
Density: Medium
Risk score: 42
Accident or Breakdown Detected
Technology Stack
Python 3.10+
Ultralytics YOLOv8
OpenCV
NumPy
Installation
1. Clone the repository
git clone https://github.com/BerkeTozkoparam/traffic-analysis-yolo.git
cd traffic-analysis-yolo
2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Usage
Place your input video file in the project directory, then run:
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
Contains structured analytics for ML pipelines:
frame_index	vehicle_count	occupancy_ratio	avg_speed_pixels	risk_score	event_flag
132	5	0.11	3.2	37	0
133	5	0.11	0.4	35	1
event_flag = 1 indicates detected accident/breakdown.
Configuration
All configuration is at the top of main.py:
CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]
STATIONARY_SPEED_THRESHOLD = 1.0
STATIONARY_FRAMES_REQUIRED = 10
Roadmap / Future Enhancements
Planned improvements:
Lane-specific vehicle counting
Actual speed estimation using homography calibration
Multi-camera fusion
Tracking-based ID persistence
Alerts over web dashboard
ROS integration for ITS systems
ONNX / TensorRT acceleration
License
You can choose a license depending on usage:
MIT for open research / demos
Apache-2.0 if you need patent inclusion
Proprietary if you want to keep closed-source
Author
Berke Baran Tozkoparan
AI / Computer Vision / Industrial Engineering projects
Contributions
Pull requests, feature suggestions and optimization ideas are welcome.
