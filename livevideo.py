import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

# ==============================
# Load Model & Feature Extractor
# ==============================
try:
    clf = joblib.load("models/stanford40_activity_clf.pkl")
    feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128,128,3))
except:
    print("Warning: Model files not found. Please ensure models/stanford40_activity_clf.pkl exists.")
    clf = None
    feature_extractor = None

# ==============================
# Custom Styled Widgets
# ==============================
class StyledButton(QPushButton):
    def __init__(self, text, primary_color="#3498db", hover_color="#2980b9", size="medium"):
        super().__init__(text)
        self.primary_color = primary_color
        self.hover_color = hover_color
        self.size = size
        self.setup_style()
        
    def setup_style(self):
        if self.size == "large":
            self.setFixedSize(200, 60)
            font_size = "18px"
            padding = "15px"
        elif self.size == "medium":
            self.setFixedSize(150, 50)
            font_size = "16px"
            padding = "12px"
        else:
            self.setFixedSize(120, 40)
            font_size = "14px"
            padding = "8px"
            
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self.primary_color}, stop:1 {self.hover_color});
                border: none;
                border-radius: 25px;
                color: white;
                font-size: {font_size};
                font-weight: bold;
                padding: {padding};
                text-align: center;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self.hover_color}, stop:1 {self.primary_color});
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                background: {self.hover_color};
                transform: translateY(1px);
            }}
        """)

class StyledLabel(QLabel):
    def __init__(self, text="", style_type="normal"):
        super().__init__(text)
        self.style_type = style_type
        self.setup_style()
        
    def setup_style(self):
        if self.style_type == "title":
            self.setStyleSheet("""
                QLabel {
                    font-size: 48px;
                    font-weight: bold;
                    color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #2E86C1, stop:1 #8E44AD);
                    text-align: center;
                    padding: 20px;
                }
            """)
        elif self.style_type == "subtitle":
            self.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    color: #34495E;
                    text-align: center;
                    padding: 15px;
                }
            """)
        elif self.style_type == "status":
            self.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    color: #2C3E50;
                    text-align: center;
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 15px;
                    border: 2px solid #BDC3C7;
                }
            """)

class VideoFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setup_style()
        
    def setup_style(self):
        self.setStyleSheet("""
            QFrame {
                border: 3px solid qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3498db, stop:1 #8E44AD);
                border-radius: 20px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ECF0F1, stop:1 #BDC3C7);
                padding: 10px;
            }
        """)
        
        # Add shadow effect
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(5, 5)
        self.setGraphicsEffect(shadow)

class DashboardPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setup_style()
        
    def setup_style(self):
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 152, 219, 0.1), stop:1 rgba(142, 68, 173, 0.1));
                border: 2px solid rgba(52, 152, 219, 0.3);
                border-radius: 25px;
                padding: 20px;
            }
        """)
        
        # Add shadow effect
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(52, 152, 219, 40))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

# ==============================
# Live Video Page
# ==============================
class LiveVideoPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.active_count = 0
        self.idle_count = 0
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Left panel - Video and controls
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        
        # Video section
        video_label = StyledLabel("🎥 Live Activity Monitoring", "subtitle")
        video_label.setAlignment(Qt.AlignCenter)
        
        self.video_container = VideoFrame()
        video_layout = QVBoxLayout(self.video_container)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #BDC3C7;
                border-radius: 15px;
                background: #2C3E50;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        
        # Prediction display
        self.pred_label = QLabel("⏳ Initializing...")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #34495E;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 20px;
                padding: 15px;
                border: 3px solid #BDC3C7;
            }
        """)
        
        # Back button
        self.back_btn = StyledButton("🏠 Back to Home", "#95A5A6", "#7F8C8D", "medium")
        self.back_btn.clicked.connect(self.parent.show_home)
        
        left_panel.addWidget(video_label)
        left_panel.addWidget(self.video_container)
        left_panel.addWidget(self.pred_label)
        left_panel.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        
        # Right panel - Dashboard
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        
        # Dashboard title
        dashboard_title = StyledLabel("📊 Real-Time Dashboard", "subtitle")
        dashboard_title.setAlignment(Qt.AlignCenter)
        
        # Dashboard container
        self.dashboard = DashboardPanel()
        dashboard_layout = QVBoxLayout(self.dashboard)
        dashboard_layout.setSpacing(15)
        
        # Status indicators
        self.alert_label = QLabel("🟢 System Ready")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #27AE60;
                background: rgba(39, 174, 96, 0.1);
                border: 2px solid #27AE60;
                border-radius: 15px;
                padding: 12px;
            }
        """)
        
        # Statistics
        stats_container = QFrame()
        stats_container.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                padding: 15px;
                border: 2px solid #BDC3C7;
            }
        """)
        stats_layout = QVBoxLayout(stats_container)
        
        self.stats_label = QLabel("📈 Activity Statistics")
        self.stats_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50; text-align: center;")
        
        self.active_stats = QLabel("🟢 Active: 0")
        self.active_stats.setStyleSheet("font-size: 18px; font-weight: bold; color: #27AE60; text-align: center;")
        
        self.idle_stats = QLabel("🔴 Idle: 0")
        self.idle_stats.setStyleSheet("font-size: 18px; font-weight: bold; color: #E74C3C; text-align: center;")
        
        stats_layout.addWidget(self.stats_label)
        stats_layout.addWidget(self.active_stats)
        stats_layout.addWidget(self.idle_stats)
        
        # Add widgets to dashboard
        dashboard_layout.addWidget(self.alert_label)
        dashboard_layout.addWidget(stats_container)
        dashboard_layout.addStretch()
        
        right_panel.addWidget(dashboard_title)
        right_panel.addWidget(self.dashboard)
        
        # Assemble main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

        # Camera Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.video_label.setText("📷 Camera Not Available")
            self.video_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #BDC3C7;
                    border-radius: 15px;
                    background: #34495E;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                }
            """)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return

        # Preprocess and predict
        if clf and feature_extractor:
            try:
                img = cv2.resize(frame, (128, 128))
                x = np.expand_dims(img.astype("float32"), axis=0)
                x = preprocess_input(x)
                features = feature_extractor.predict(x, verbose=0)
                pred = clf.predict(features)[0]
                
                # Update dashboard
                if pred == 1:  # Active
                    self.pred_label.setText("🎯 ACTIVE WORKER ✅")
                    self.pred_label.setStyleSheet("""
                        QLabel {
                            font-size: 24px;
                            font-weight: bold;
                            color: #27AE60;
                            background: rgba(39, 174, 96, 0.1);
                            border-radius: 20px;
                            padding: 15px;
                            border: 3px solid #27AE60;
                        }
                    """)
                    self.alert_label.setText("🟢 Status: Normal - Worker Active")
                    self.alert_label.setStyleSheet("""
                        QLabel {
                            font-size: 18px;
                            font-weight: bold;
                            color: #27AE60;
                            background: rgba(39, 174, 96, 0.1);
                            border: 2px solid #27AE60;
                            border-radius: 15px;
                            padding: 12px;
                        }
                    """)
                    self.active_count += 1
                else:  # Idle
                    self.pred_label.setText("⚠️ IDLE WORKER DETECTED ❌")
                    self.pred_label.setStyleSheet("""
                        QLabel {
                            font-size: 24px;
                            font-weight: bold;
                            color: #E74C3C;
                            background: rgba(231, 76, 60, 0.1);
                            border-radius: 20px;
                            padding: 15px;
                            border: 3px solid #E74C3C;
                        }
                    """)
                    self.alert_label.setText("🔴 ALERT: Worker Idle - Action Required!")
                    self.alert_label.setStyleSheet("""
                        QLabel {
                            font-size: 18px;
                            font-weight: bold;
                            color: #E74C3C;
                            background: rgba(231, 76, 60, 0.1);
                            border: 2px solid #E74C3C;
                            border-radius: 15px;
                            padding: 12px;
                        }
                    """)
                    self.idle_count += 1
                
                # Update statistics
                self.active_stats.setText(f"🟢 Active: {self.active_count}")
                self.idle_stats.setText(f"🔴 Idle: {self.idle_count}")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                self.pred_label.setText("❌ Prediction Error")
                self.pred_label.setStyleSheet("""
                    QLabel {
                        font-size: 24px;
                        font-weight: bold;
                        color: #E74C3C;
                        background: rgba(231, 76, 60, 0.1);
                        border-radius: 20px;
                        padding: 15px;
                        border: 3px solid #E74C3C;
                    }
                """)

        # Show Video
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
