"""
Camera User Interface for Z-CAM Emotion Detection System - FIXED VERSION
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout,
    QVBoxLayout, QFrame, QSizePolicy, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
import cv2
import sys


# ============================================================================
# MAIN APPLICATION WINDOW
# ============================================================================

class UISignals(QObject):
    people_updated = pyqtSignal(list)


class CameraUI(QWidget):
    
    def __init__(self, frame_bridge):
        super().__init__()
        
        self.bridge = frame_bridge
        self.bridge.set_frame_callback(self.update_display_frame)
        
        # ====================================================================
        # WINDOW CONFIGURATION
        # ====================================================================
        self.setWindowTitle("Z-CAM Emotion Detection System")
        self.setMinimumWidth(320)
        self.setStyleSheet("""
            QWidget { background-color: white; color: black; }
            QLabel { color: black; font-size: 16px; }
        """)
        
        self.signals = UISignals()
        self.signals.people_updated.connect(self.update_people)
        
        # ====================================================================
        # TOP AREA WITH LOGO
        # ====================================================================
        self.top_area = QFrame()
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry()
        top_height = int(screen_geom.height() * 0.18)
        self.top_area.setFixedHeight(top_height)

        top_layout = QHBoxLayout(self.top_area)
        top_layout.setContentsMargins(20, 0, 20, 0)

        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(left_spacer)

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

        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(right_spacer)

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
        """)
        self.info_box.setMinimumHeight(200)
        self.info_box.setMinimumWidth(400) 

        self.info_layout = QGridLayout(self.info_box)
        self.info_layout.setSpacing(6)
        self.info_layout.setContentsMargins(6, 6, 6, 6)

        self.person_rows = []

        # ====================================================================
        # MAIN WINDOW LAYOUT ASSEMBLY
        # ====================================================================
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.top_area)
        central_layout.addWidget(self.camera_container, stretch=1)
        central_layout.addWidget(self.info_box)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(central_layout, stretch=1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame_from_bridge)
        self.timer.start(30)

        self.current_frame = None
        self.no_signal_text = "ŁĄCZENIE Z KAMERĄ..."

    # ========================================================================
    # FRAME UPDATE FROM BRIDGE
    # ========================================================================
    
    def update_display_frame(self, frame):
        self.current_frame = frame
    
    def update_frame_from_bridge(self):
        if self.current_frame is None:
            self.show_no_signal()
            return
        
        self.display_frame(self.current_frame)
    
    def show_no_signal(self):
        w = self.cam_label.width()
        h = self.cam_label.height()
        
        if w <= 0 or h <= 0:
            return
            
        black_frame = QImage(w, h, QImage.Format_RGB888)
        black_frame.fill(Qt.black)
        
        pixmap = QPixmap.fromImage(black_frame)
        painter = QPainter(pixmap)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 14))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, self.no_signal_text)
        painter.end()
        
        self.cam_label.setPixmap(pixmap)
    
    def display_frame(self, frame):
        w = self.cam_label.width()
        h = self.cam_label.height()

        if w <= 0 or h <= 0:
            return

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
        
        display_frame = cropped.copy()
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        image = QImage(resized.data, w, h, w * 3, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(image))

    # ========================================================================
    # PEOPLE DISPLAY
    # ========================================================================
    
    @pyqtSlot(list)
    def update_people(self, people):
        for box in self.person_rows:
            self.info_layout.removeWidget(box)
            box.deleteLater()
        self.person_rows.clear()

        if not people:
            empty_label = QLabel("No people detected")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: white; font-size: 14px; font-style: italic;")
            self.info_layout.addWidget(empty_label, 0, 0, 1, 2)
            self.person_rows.append(empty_label)
            return

        font_size = max(10, 20 - len(people) * 2)
        max_cols = 2

        for idx, person in enumerate(people):
            row = idx // max_cols
            col = idx % max_cols

            box = QFrame()
            box.setStyleSheet("""
                QFrame {
                    background-color: rgba(255, 255, 255, 0.2);
                    border-radius: 6px;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    padding: 4px;
                }
            """)
            
            v = QVBoxLayout(box)
            v.setContentsMargins(4, 4, 4, 4)
            v.setSpacing(2)

            if 'id' in person:
                person_id = person['id']
                emotion = person.get('emotion', 'Unknown')
                confidence = person.get('confidence', 0.0)
            else:
                person_id = idx + 1
                emotion = 'Unknown'
                confidence = 0.0

            lbl_person = QLabel(f"Person: {person_id}")
            
            if confidence > 0:
                lbl_emotion = QLabel(f"Emotion: {emotion} ({confidence*100:.1f}%)")
            else:
                lbl_emotion = QLabel(f"Emotion: {emotion}")

            emotion_colors = {
                'Happy': "#51FA51",
                'Sad': "#5ED0FD", 
                'Angry': "#FA583C",
                'Surprise': "#F5A802",
                'Fear': '#9370DB', 
                'Disgust': '#32CD32', 
                'Neutral': "#555555", 
                'Unknown': "#FFFFFF" 
            }
            
            emotion_color = emotion_colors.get(emotion, 'white')

            lbl_person.setAlignment(Qt.AlignCenter)
            lbl_person.setStyleSheet(f"""
                color: #FFFFFF; 
                font-size: {font_size}px; 
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            """)
            
            lbl_emotion.setAlignment(Qt.AlignCenter)
            lbl_emotion.setStyleSheet(f"""
                color: {emotion_color}; 
                font-size: {font_size}px; 
                font-weight: bold;
                padding: 0px;
                margin: 0px;
            """)

            v.addWidget(lbl_person)
            v.addWidget(lbl_emotion)

            self.info_layout.addWidget(box, row, col)
            self.person_rows.append(box)
    # ========================================================================
    # RESIZE HANDLING
    # ========================================================================
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_camera_size()

    def update_camera_size(self):
        container = self.camera_container
        avail_width = container.width()
        avail_height = container.height()

        reserved_height = self.top_area.height() + self.info_box.height() + 100
        max_cam_height = self.height() - reserved_height

        if max_cam_height < 100 or avail_width < 100:
            return

        target_w = avail_width
        target_h = int(target_w * 9 / 16)

        if target_h > max_cam_height:
            target_h = max_cam_height
            target_w = int(target_h * 16 / 9)

        target_w = max(target_w, 320)
        target_h = max(target_h, 180)

        self.cam_label.setFixedSize(target_w, target_h)

    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    def closeEvent(self, event):
        """Handle application shutdown"""
        print("[UI] Window closed")
        super().closeEvent(event)


# ============================================================================
# TEST MODE
# ============================================================================

if __name__ == "__main__":
    """Testuj UI bez AutoAdjust"""
    print("[UI TEST] Running UI in standalone test mode")
    
    class TestBridge:
        def __init__(self):
            self.current_frame = None
        
        def set_frame_callback(self, callback):
            self.frame_callback = callback
        
        def update_frame(self, frame, people):
            self.current_frame = frame
    
    app = QApplication(sys.argv)
    bridge = TestBridge()
    window = CameraUI(bridge)
    window.resize(1000, 700)
    window.show()
    
    # ========================================================================
    # UPDATED TEST CASES
    # ========================================================================
    import threading
    import time
    import numpy as np
    
    def send_test_data():
        """Wysyłaj testowe dane"""
        time.sleep(1)
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:300, 100:300] = [0, 0, 255]
        bridge.update_frame(test_frame, [])
        
        test_cases = [
            [],
            
            [{'id': 1, 'emotion': 'Happy', 'confidence': 0.85}],
            
            [
                {'id': 1, 'emotion': 'Happy', 'confidence': 0.92},
                {'id': 2, 'emotion': 'Sad', 'confidence': 0.78}
            ],
            
            [
                {'id': 1, 'emotion': 'Angry', 'confidence': 0.65},
                {'id': 2, 'emotion': 'Surprise', 'confidence': 0.88},
                {'id': 3, 'emotion': 'Neutral', 'confidence': 0.95}
            ],
            
            [
                {'id': 1, 'emotion': 'Happy', 'confidence': 0.91},
                {'id': 2, 'emotion': 'Sad', 'confidence': 0.82},
                {'id': 3, 'emotion': 'Angry', 'confidence': 0.73},
                {'id': 4, 'emotion': 'Fear', 'confidence': 0.68}
            ],
            
            [
                {'id': 1, 'emotion': 'Happy', 'confidence': 0.94},
                {'id': 2, 'emotion': 'Sad', 'confidence': 0.76},
                {'id': 3, 'emotion': 'Angry', 'confidence': 0.81},
                {'id': 4, 'emotion': 'Surprise', 'confidence': 0.89},
                {'id': 5, 'emotion': 'Fear', 'confidence': 0.72},
                {'id': 6, 'emotion': 'Neutral', 'confidence': 0.65}
            ]
        ]
        
        for i, people in enumerate(test_cases):
            print(f"\n[TEST] Case {i+1}: Sending {len(people)} people")
            for person in people:
                print(f"  - Person {person['id']}: {person['emotion']} ({person['confidence']*100:.1f}%)")
            
            window.signals.people_updated.emit(people)
            time.sleep(3)
        
        print("\n[TEST] All test cases completed!")
        window.signals.people_updated.emit([])
    
    test_thread = threading.Thread(target=send_test_data, daemon=True)
    test_thread.start()
    
    sys.exit(app.exec_())