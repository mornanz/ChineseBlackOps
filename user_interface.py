"""
Camera User Interface for Z-CAM Emotion Detection System
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QFrame, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect
from PyQt5.QtGui import QImage, QPixmap
import cv2
import sys


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class CameraUI(QWidget):
    """
    Responsive camera application interface with real-time video display
    and emotion detection visualization.
    """
    
    def __init__(self):
        """Initialize UI components, layouts, and camera capture"""
        super().__init__()
        
        # ====================================================================
        # WINDOW CONFIGURATION
        # ====================================================================
        self.setWindowTitle("Z-CAM Emotion Detection System")
        self.setMinimumWidth(650)
        self.setStyleSheet("""
            QWidget { background-color: white; color: black; }
            QLabel { color: black; font-size: 16px; }
            QPushButton { 
                background-color: #0078d7; 
                border: none; 
                padding: 10px; 
                border-radius: 6px; 
                color: black; 
                font-size: 14px; 
            }
            QPushButton:hover { background-color: #005ea6; }
        """)

        # ====================================================================
        # TOP AREA WITH LOGO AND CONTROLS
        # ====================================================================
        self.top_area = QFrame()
        
        # Dynamically set height based on screen resolution
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry()
        top_height = int(screen_geom.height() * 0.18)
        self.top_area.setFixedHeight(top_height)

        top_layout = QHBoxLayout(self.top_area)
        top_layout.setContentsMargins(20, 0, 20, 0)

        # Left spacer for flexible layout
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(left_spacer)

        # Central logo display
        self.logo = QLabel()
        self.logo.setAlignment(Qt.AlignCenter)
        logo_path = "./logo_images/witlogo1.png"
        pixmap = QPixmap(logo_path)

        if pixmap.isNull():
            self.logo.setText("[UI ERROR] LOGO NOT FOUND")
            self.logo.setStyleSheet("color: red; font-size: 14px;")
        else:
            target_height = int(top_height * 1.5)
            scaled_pixmap = pixmap.scaled(
                target_height, target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.logo.setPixmap(scaled_pixmap)

        top_layout.addWidget(self.logo, alignment=Qt.AlignCenter)

        # Right section with settings button
        right_wrapper = QWidget()
        right_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_layout = QHBoxLayout(right_wrapper)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_settings = QPushButton("⚙️ Settings")
        self.btn_settings.setFixedWidth(100)
        self.btn_settings.clicked.connect(self.toggle_settings)
        
        right_layout.addStretch()
        right_layout.addWidget(self.btn_settings)
        top_layout.addWidget(right_wrapper)

        # ====================================================================
        # CENTRAL CAMERA DISPLAY AREA
        # ====================================================================
        self.camera_container = QWidget()
        self.camera_container_layout = QVBoxLayout(self.camera_container)
        self.camera_container_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_container_layout.setSpacing(0)

        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("background-color: #111; border: 3px solid #0078d7;")
        self.cam_label.setMinimumSize(320, 180)

        # Center camera display using stretch spacers
        self.camera_container_layout.addStretch(1)
        self.camera_container_layout.addWidget(self.cam_label, alignment=Qt.AlignCenter)
        self.camera_container_layout.addStretch(1)

        # ====================================================================
        # EMOTION DETECTION STATUS DISPLAY
        # ====================================================================
        self.info_box = QFrame()
        self.info_box.setStyleSheet("""
            background-color: #0078d7; 
            color: white; 
            border-radius: 8px; 
            padding: 10px;
        """)
        self.info_box.setFixedHeight(100)
        
        info_layout = QVBoxLayout(self.info_box)
        self.label_person = QLabel("Person: -")
        self.label_emotion = QLabel("Emotion: -")
        
        for lbl in (self.label_person, self.label_emotion):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
            info_layout.addWidget(lbl)

        # ====================================================================
        # SETTINGS SIDEBAR PANEL
        # ====================================================================
        self.settings_panel = QWidget()
        self.settings_panel.setMaximumWidth(0)
        self.settings_panel.setVisible(False)
        self.settings_panel.setStyleSheet("""
            background-color: white; 
            border-left: 3px solid #0078d7;
        """)
        
        settings_layout = QVBoxLayout(self.settings_panel)
        
        # Settings button definitions with action mappings
        settings_config = {
            "Exposure": self.set_exposure,
            "Capture Image": self.save_frame,
            "Exit": self.close
        }
        
        for text, action in settings_config.items():
            btn = QPushButton(text)
            btn.setFixedWidth(200)
            btn.clicked.connect(action)
            settings_layout.addWidget(btn, alignment=Qt.AlignCenter)
            
        settings_layout.addStretch(1)

        # ====================================================================
        # MAIN WINDOW LAYOUT ASSEMBLY
        # ====================================================================
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.top_area)
        central_layout.addWidget(self.camera_container, stretch=1)
        central_layout.addWidget(self.info_box)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(central_layout, stretch=1)
        main_layout.addWidget(self.settings_panel)

        # ====================================================================
        # VIDEO CAPTURE INITIALIZATION
        # ====================================================================
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.current_frame = None
        self.settings_anim = None

    # ========================================================================
    # WINDOW RESIZE HANDLING
    # ========================================================================
    
    def resizeEvent(self, event):
        """
        Handle window resize events to maintain camera display proportions
        
        Parameters:
            event: QResizeEvent containing new window dimensions
        """
        super().resizeEvent(event)
        self.update_camera_size()

    def update_camera_size(self):
        """Recalculate camera display size based on available window space"""
        container = self.camera_container
        avail_width = container.width()
        avail_height = container.height()

        # Calculate reserved space for UI elements
        reserved_height = self.top_area.height() + self.info_box.height() + 100
        max_cam_height = self.height() - reserved_height

        if max_cam_height < 100 or avail_width < 100:
            return

        # Maintain 16:9 aspect ratio
        target_w = avail_width
        target_h = int(target_w * 9 / 16)

        if target_h > max_cam_height:
            target_h = max_cam_height
            target_w = int(target_h * 16 / 9)

        # Apply minimum size constraints
        target_w = max(target_w, 320)
        target_h = max(target_h, 180)

        self.cam_label.setFixedSize(target_w, target_h)

    # ========================================================================
    # SETTINGS PANEL CONTROL
    # ========================================================================
    
    def toggle_settings(self):
        """Animate settings panel slide-in/out with smooth transition"""
        target_width = 220 if not self.settings_panel.isVisible() else 0
        self.settings_panel.setVisible(True)

        self.settings_anim = QPropertyAnimation(self.settings_panel, b"maximumWidth")
        self.settings_anim.setDuration(300)
        self.settings_anim.setStartValue(self.settings_panel.width())
        self.settings_anim.setEndValue(target_width)
        
        if target_width == 0:
            self.settings_anim.finished.connect(
                lambda: self.settings_panel.setVisible(False)
            )
            
        self.settings_anim.start()

        # Update camera size after animation completes
        QTimer.singleShot(320, self.update_camera_size)

    # ========================================================================
    # SETTINGS ACTIONS
    # ========================================================================
    
    def set_exposure(self):
        """Placeholder for exposure control implementation"""
        print("[UI] Exposure settings panel requested")

    def save_frame(self):
        """Capture and save current video frame to disk"""
        if self.current_frame is not None:
            cv2.imwrite("capture.png", self.current_frame)
            print("[UI] Frame saved as capture.png")

    # ========================================================================
    # VIDEO FRAME PROCESSING
    # ========================================================================
    
    def update_frame(self):
        """Capture and display video frame with aspect ratio preservation"""
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        # Get current display dimensions
        w = self.cam_label.width()
        h = self.cam_label.height()

        if w <= 0 or h <= 0:
            return

        # Crop to maintain 16:9 aspect ratio
        frame_h, frame_w = frame.shape[:2]
        target_ratio = w / h
        frame_ratio = frame_w / frame_h

        if frame_ratio > target_ratio:
            new_w = int(frame_h * target_ratio)
            start_x = (frame_w - new_w) // 2
            cropped = frame[:, start_x:start_x + new_w]
        else:
            new_h = int(frame_w / target_ratio)
            start_y = (frame_h - new_h) // 2
            cropped = frame[start_y:start_y + new_h, :]

        # Convert and resize for display
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        image = QImage(resized.data, w, h, w * 3, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(image))

        # TODO: Integrate with emotion detection model
        self.label_person.setText("Person: John Doe")
        self.label_emotion.setText("Emotion: Smile 😊")

    # ========================================================================
    # CLEANUP AND SHUTDOWN
    # ========================================================================
    
    def closeEvent(self, event):
        """
        Handle application shutdown with resource cleanup
        
        Parameters:
            event: QCloseEvent triggered by window close
        """
        self.cap.release()
        print("[UI] Camera resources released")
        super().closeEvent(event)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Main application entry point with error handling"""
    app = QApplication(sys.argv)
    window = CameraUI()
    window.showMaximized()
    sys.exit(app.exec_())