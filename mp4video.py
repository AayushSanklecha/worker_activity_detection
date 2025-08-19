import cv2
import numpy as np
import joblib
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QFileDialog,
    QProgressBar, QSlider, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
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
# Video Processing Thread
# ==============================
class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray, int, int)  # frame, prediction, frame_number
    processing_complete = pyqtSignal()
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                try:
                    # Preprocess frame
                    img = cv2.resize(frame, (128, 128))
                    x = np.expand_dims(img.astype("float32"), axis=0)
                    x = preprocess_input(x)
                    
                    # Extract features and predict
                    if clf and feature_extractor:
                        features = feature_extractor.predict(x, verbose=0)
                        pred = clf.predict(features)[0]
                    else:
                        pred = 0  # Default to idle if model not available
                    
                    # Emit processed frame
                    self.frame_processed.emit(frame, pred, frame_count)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    self.frame_processed.emit(frame, 0, frame_count)
            
            frame_count += 1
            
        cap.release()
        self.processing_complete.emit()

# ==============================
# MP4 Video Page
# ==============================
class MP4VideoPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
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
        video_label = StyledLabel("📁 MP4 Video Analysis", "subtitle")
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
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("📁 Upload an MP4 video to begin analysis")
        
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        
        # Video controls
        controls_layout = QHBoxLayout()
        
        self.upload_btn = StyledButton("📁 Upload Video", "#F39C12", "#E67E22", "medium")
        self.upload_btn.clicked.connect(self.upload_video)
        
        self.play_btn = StyledButton("▶️ Play", "#27AE60", "#229954", "medium")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = StyledButton("⏹️ Stop", "#E74C3C", "#C0392B", "medium")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.upload_btn)
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        
        # Progress bar and slider
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #8E44AD);
                border-radius: 8px;
            }
        """)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #BDC3C7;
                height: 8px;
                background: #ECF0F1;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.frame_slider)
        
        # Prediction display
        self.pred_label = QLabel("⏳ Waiting for video...")
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
        left_panel.addLayout(controls_layout)
        left_panel.addLayout(progress_layout)
        left_panel.addWidget(self.pred_label)
        left_panel.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        
        # Right panel - Dashboard
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        
        # Dashboard title
        dashboard_title = StyledLabel("📊 Video Analysis Dashboard", "subtitle")
        dashboard_title.setAlignment(Qt.AlignCenter)
        
        # Dashboard container
        self.dashboard = DashboardPanel()
        dashboard_layout = QVBoxLayout(self.dashboard)
        dashboard_layout.setSpacing(15)
        
        # Status indicators
        self.alert_label = QLabel("🟢 Ready for Video Upload")
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
        
        self.stats_label = QLabel("📈 Analysis Statistics")
        self.stats_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50; text-align: center;")
        
        self.active_stats = QLabel("🟢 Active: 0")
        self.active_stats.setStyleSheet("font-size: 18px; font-weight: bold; color: #27AE60; text-align: center;")
        
        self.idle_stats = QLabel("🔴 Idle: 0")
        self.idle_stats.setStyleSheet("font-size: 18px; font-weight: bold; color: #E74C3C; text-align: center;")
        
        self.total_frames_label = QLabel("📹 Total Frames: 0")
        self.total_frames_label.setStyleSheet("font-size: 16px; color: #34495E; text-align: center;")
        
        stats_layout.addWidget(self.stats_label)
        stats_layout.addWidget(self.active_stats)
        stats_layout.addWidget(self.idle_stats)
        stats_layout.addWidget(self.total_frames_label)
        
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

        # Video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select MP4 Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            self.load_video()
            self.upload_btn.setText("✅ Video Loaded!")
            self.upload_btn.setStyleSheet("""
                QPushButton {
                    background: #27AE60;
                    border: none;
                    border-radius: 25px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 12px;
                }
            """)
            # Reset button after 2 seconds
            QTimer.singleShot(2000, lambda: self.reset_upload_button())

    def reset_upload_button(self):
        self.upload_btn.setText("📁 Upload Video")
        self.upload_btn.setup_style()

    def load_video(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                # Update UI
                self.total_frames_label.setText(f"📹 Total Frames: {self.total_frames}")
                self.frame_slider.setMaximum(self.total_frames - 1)
                self.progress_bar.setMaximum(self.total_frames - 1)
                
                # Enable controls
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.frame_slider.setEnabled(True)
                
                # Show first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                self.alert_label.setText("🟢 Video Loaded - Ready to Play")
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

    def toggle_play(self):
        if not self.is_playing:
            self.play_video()
        else:
            self.pause_video()

    def play_video(self):
        if self.cap and self.cap.isOpened():
            self.is_playing = True
            self.play_btn.setText("⏸️ Pause")
            self.timer.start(int(1000 / self.fps))
            self.alert_label.setText("🟢 Playing Video")
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

    def pause_video(self):
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.timer.stop()

    def stop_video(self):
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.timer.stop()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.frame_slider.setValue(0)
            self.progress_bar.setValue(0)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)

    def seek_frame(self, frame_number):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.analyze_frame(frame)

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
                self.progress_bar.setValue(self.current_frame)
                self.display_frame(frame)
                self.analyze_frame(frame)
                
                if self.current_frame >= self.total_frames - 1:
                    self.stop_video()
            else:
                self.stop_video()

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def analyze_frame(self, frame):
        if clf and feature_extractor:
            try:
                # Preprocess frame
                img = cv2.resize(frame, (128, 128))
                x = np.expand_dims(img.astype("float32"), axis=0)
                x = preprocess_input(x)
                
                # Extract features and predict
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

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.timer:
            self.timer.stop()
