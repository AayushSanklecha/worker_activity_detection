import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QFileDialog, QStackedWidget,
    QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtCore import QTimer, Qt
from livevideo import LiveVideoPage
from mp4video import MP4VideoPage

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
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(5, 5)
        self.setGraphicsEffect(shadow)

# ==============================
# Main App with Page Switching
# ==============================
class ActiTrackMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ActiTrack - AI-Powered Worker Activity Recognition")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Set application icon and style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ECF0F1, stop:0.5 #F8F9FA, stop:1 #E8F4FD);
            }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Pages
        self.home_page = HomePage(self)
        self.live_video_page = LiveVideoPage(self)
        self.mp4_video_page = MP4VideoPage(self)

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.live_video_page)
        self.stacked_widget.addWidget(self.mp4_video_page)

        self.stacked_widget.setCurrentWidget(self.home_page)

    def show_live_video(self):
        self.stacked_widget.setCurrentWidget(self.live_video_page)

    def show_mp4_video(self):
        self.stacked_widget.setCurrentWidget(self.mp4_video_page)

    def show_home(self):
        self.stacked_widget.setCurrentWidget(self.home_page)

# ==============================
# Page 1: Home Page
# ==============================
class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        # Main layout with gradient background
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header section
        header_layout = QVBoxLayout()
        
        # Main title with gradient
        title = StyledLabel("⚡ ActiTrack ⚡", "title")
        title.setAlignment(Qt.AlignCenter)
        
        # Subtitle
        subtitle = StyledLabel("AI-Powered Worker Activity Recognition System", "subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        
        # Video preview section
        preview_section = QVBoxLayout()
        preview_label = StyledLabel("📹 Live Camera Preview", "subtitle")
        preview_label.setAlignment(Qt.AlignCenter)
        
        # Video frame container
        self.video_container = VideoFrame()
        video_layout = QVBoxLayout(self.video_container)
        
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 480)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px solid #BDC3C7;
                border-radius: 15px;
                background: #2C3E50;
            }
        """)
        self.preview_label.setAlignment(Qt.AlignCenter)
        
        video_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        preview_section.addWidget(preview_label)
        preview_section.addWidget(self.video_container)
        
        # Buttons section
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(30)
        buttons_layout.setAlignment(Qt.AlignCenter)
        
        self.start_btn = StyledButton("🚀 Live Tracking", "#27AE60", "#229954", "large")
        self.start_btn.clicked.connect(self.parent.show_live_video)
        
        self.upload_btn = StyledButton("📁 Upload Video", "#F39C12", "#E67E22", "large")
        self.upload_btn.clicked.connect(self.parent.show_mp4_video)
        
        self.exit_btn = StyledButton("🚪 Exit", "#E74C3C", "#C0392B", "large")
        self.exit_btn.clicked.connect(QApplication.instance().quit)
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.upload_btn)
        buttons_layout.addWidget(self.exit_btn)
        
        # Assemble main layout
        main_layout.addLayout(header_layout)
        main_layout.addLayout(preview_section)
        main_layout.addLayout(buttons_layout)
        
        # Add spacer at bottom
        main_layout.addStretch()
        self.setLayout(main_layout)

        # Webcam Preview Timer
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(50)

    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            # Show placeholder when camera not available
            self.preview_label.setText("📷 Camera Not Available")
            self.preview_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #BDC3C7;
                    border-radius: 15px;
                    background: #34495E;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                }
            """)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()

# ==============================
# Run Application
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = ActiTrackMain()
    window.show()
    sys.exit(app.exec_())
