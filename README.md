# ActiTrack — Real-Time Worker Activity Recognition

ActiTrack is an AI-powered system that classifies worker activity (Active vs Idle) from video streams in real time.  
It uses motion analysis inside a selected region of interest (ROI) to provide live monitoring, idle alerts, and productivity insights.

---

## 🚀 Features
- Real-time detection of **Active** vs **Idle** worker states
- Selectable ROI (region of interest) for each workstation
- Idle timer and configurable alert thresholds
- Live dashboard with activity stats
- Works with webcam or uploaded videos
- Lightweight, CPU-friendly

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **OpenCV** for video + motion detection
- **Streamlit** for interactive dashboard UI
- **NumPy, Pandas** for data handling
- *(Optional)* MediaPipe + scikit-learn for pose-based ML
