"""
Real-time ISO Optimization with Facial Emotion Detection
Author: [Your Name]
Date: [Date]
Description: Implements adaptive ISO control based on image histogram analysis
             with concurrent facial emotion detection using YOLOv8.
License: [Your License]
"""

import numpy as np
import cv2 as cv
import asyncio
import os
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from camera_control import create_camera_ctrl  # Camera control module

# ============================================================================
# CAMERA INTERFACE ABSTRACTION
# ============================================================================

class CameraStub:
    """Fallback camera interface for offline testing"""
    def __init__(self):
        self.iso_settings = [500, 640, 1600, 3200, 6400, 12800, 25600, 51200]
        self._iso = 1600

    def get_iso(self):
        """Return current ISO setting"""
        return self._iso

    def set_iso(self, value):
        """Set ISO value if within allowed settings"""
        if value not in self.iso_settings:
            print(f"[CAMERA] ISO {value} not in settings, keeping {self._iso}")
            return
        print(f"[CAMERA] Changing ISO {self._iso} -> {value}")
        self._iso = value


# ============================================================================
# FRAME CAPTURE AND PROCESSING
# ============================================================================

class AutoAdjust:
    """Main capture class handling real-time frame processing and emotion detection"""
    
    def __init__(self, queue: asyncio.Queue, camera, save_frames=False, display_frames=True):
        self.loopState = False
        self.cap = cv.VideoCapture(0, cv.CAP_ANY)
        self.queue = queue
        self.camera = camera
        self.save_frames = save_frames
        self.display_frames = display_frames
        self.frame_count = 0
        self.last_display_time = 0
        self.display_interval = 0.2  # ~5 FPS display rate

        if save_frames:
            os.makedirs("auto_adjust_frames", exist_ok=True)

        # Initialize YOLOv8 emotion detection model
        print("[INIT] Loading YOLO model best.pt...")
        self.yolo = YOLO("best.pt")
        print("[INIT] YOLO classes loaded")

        # Initialize Haar cascade for face detection
        cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print(f"[WARN] Could not load Haar cascade from {cascade_path}")
            self.face_cascade = None

    def CaptureFrame(self):
        """Capture single frame from video source"""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def HistogramFromFrame(self, frame):
        """Convert frame to grayscale and compute histogram"""
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        return hist, img

    def run_emotion_detection(self, frame, gray_img):
        """Detect faces and classify emotions using YOLOv8"""
        if self.face_cascade is None:
            return

        faces = self.face_cascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            # YOLO emotion prediction
            results = self.yolo.predict(
                face_img,
                conf=0.25,
                iou=0.45,
                imgsz=320,
                verbose=False
            )
            result = results[0]

            if len(result.boxes) > 0:
                box = result.boxes[0]
                cls_id = int(box.cls[0])
                label = self.yolo.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])

                # Draw detection bounding box and label
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x, max(y - 10, 20)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                # Draw face rectangle without emotion label
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def DisplayFrameWithHistogram(self, frame, gray_img, hist, iso, mean, median):
        """Display processed frame with histogram overlay and statistics"""
        try:
            display_frame = cv.resize(frame, (640, 480))
            hist_display = np.zeros((200, 640, 3), dtype=np.uint8)

            # Render histogram visualization
            hist_flat = hist.flatten()
            if np.sum(hist_flat) > 0:
                hist_normalized = cv.normalize(hist_flat, None, 0, 200, cv.NORM_MINMAX)
                bin_width = 640 // 256
                for i in range(255):
                    x1 = i * bin_width
                    x2 = (i + 1) * bin_width
                    y1 = 200 - int(hist_normalized[i])
                    y2 = 200 - int(hist_normalized[i + 1])
                    cv.line(hist_display, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Calculate marker positions
            mean_x = int((mean / 255) * 640)
            median_x = int((median / 255) * 640)
            target_x = int((100 / 255) * 640)

            # Draw statistical markers
            cv.line(hist_display, (mean_x, 0), (mean_x, 200), (0, 0, 255), 2)
            cv.line(hist_display, (median_x, 0), (median_x, 200), (255, 0, 0), 2)
            cv.line(hist_display, (target_x, 0), (target_x, 200), (0, 255, 0), 1)

            # Overlay text information
            cv.putText(display_frame, f"ISO: {iso}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_frame, f"Mean: {mean:.1f}", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(display_frame, f"Median: {median:.1f}", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(display_frame, "Target: 100", (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_frame, f"Frame: {self.frame_count}", (10, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined = np.vstack([display_frame, hist_display])
            cv.imshow("Auto Adjust + Emotions", combined)
        except Exception as e:
            print(f"[DISPLAY ERROR] {e}")

    def InterruptLoop(self):
        """Signal to stop capture loop"""
        self.loopState = False

    async def autoadjustLoop(self):
        """Main capture and processing loop"""
        try:
            self.loopState = True

            while self.loopState:
                frame = self.CaptureFrame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                hist, gray_img = self.HistogramFromFrame(frame)

                # Emotion detection on faces
                self.run_emotion_detection(frame, gray_img)

                current_iso = self.camera.get_iso() if self.camera is not None else 800

                # Queue frame data for analysis
                try:
                    self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                except asyncio.QueueFull:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                    except Exception:
                        pass

                # Display at reduced frame rate
                current_time = asyncio.get_event_loop().time()
                if self.display_frames and (current_time - self.last_display_time >= self.display_interval):
                    mean = float(np.mean(gray_img))
                    median = float(np.median(gray_img))
                    self.DisplayFrameWithHistogram(frame, gray_img, hist, current_iso, mean, median)
                    self.last_display_time = current_time

                # Keyboard controls
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.display_frames = False
                    self.loopState = False
                    cv.destroyAllWindows()
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"auto_adjust_frames/manual_frame_{timestamp}.jpg"
                    cv.imwrite(filename, frame)

                self.frame_count += 1
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[CAPTURE ERROR] {e}")
        finally:
            self.cap.release()
            cv.destroyAllWindows()


# ============================================================================
# HISTOGRAM ANALYSIS AND ISO OPTIMIZATION
# ============================================================================

class HistogramAnalyzer:
    """ISO optimization based on real-time histogram analysis"""
    
    def __init__(self, queue: asyncio.Queue, camera):
        self.queue = queue
        self.camera = camera
        self.mean_buffer = deque(maxlen=12)
        self.last_iso_change = 0
        self.iso_change_cooldown = 10
        self.target_mean = 100
        self.stable_frames = 0
        self.last_analysis_time = 0
        self.analysis_interval = 1.0
        self.iso_list = self.camera.iso_settings

    def calculate_optimal_iso(self, smooth_mean, dark_ratio, bright_ratio):
        """Determine optimal ISO based on smoothed histogram mean"""
        current_iso = self.camera.get_iso()

        try:
            current_idx = self.iso_list.index(current_iso)
        except ValueError:
            current_idx = 2

        current_zone = self._get_iso_zone(current_iso)

        # Zone determination with hysteresis thresholds
        if smooth_mean < 20:
            target_zone = 6
            hysteresis_threshold = 25
        elif smooth_mean < 45:
            target_zone = 5
            hysteresis_threshold = 40 if current_zone >= 5 else 25
        elif smooth_mean < 80:
            target_zone = 4
            hysteresis_threshold = 75 if current_zone >= 4 else 40
        elif smooth_mean < 105:
            target_zone = 3
            hysteresis_threshold = 100 if current_zone >= 3 else 75
        elif smooth_mean < 130:
            target_zone = 2
            hysteresis_threshold = 125 if current_zone >= 2 else 100
        elif smooth_mean < 160:
            target_zone = 1
            hysteresis_threshold = 150 if current_zone >= 1 else 125
        else:
            target_zone = 0
            hysteresis_threshold = 160

        # Apply hysteresis to prevent rapid oscillation
        if current_zone != target_zone:
            if (target_zone > current_zone and smooth_mean < hysteresis_threshold) or \
               (target_zone < current_zone and smooth_mean > hysteresis_threshold):
                target_zone = current_zone

        # Map zone to ISO index
        zone_to_iso = {
            6: len(self.iso_list) - 1,
            5: len(self.iso_list) - 2,
            4: len(self.iso_list) // 2,
            3: max(0, len(self.iso_list) // 4),
            2: min(4, len(self.iso_list) - 1),
            1: 1,
            0: 0
        }

        new_idx = zone_to_iso[target_zone]

        if new_idx != current_idx:
            print(f"[HYSTERESIS] mean={smooth_mean:.1f}, zone {current_zone}->{target_zone}, "
                  f"ISO {self.iso_list[current_idx]}->{self.iso_list[new_idx]}")
        else:
            print(f"[HYSTERESIS] mean={smooth_mean:.1f}, staying zone {current_zone}, ISO {self.iso_list[current_idx]}")

        return self.iso_list[new_idx]

    def _get_iso_zone(self, iso):
        """Categorize ISO value into sensitivity zones"""
        if iso >= 40000:
            return 6
        elif iso >= 16000:
            return 5
        elif iso >= 3200:
            return 4
        elif iso >= 1600:
            return 3
        elif iso >= 640:
            return 2
        elif iso >= 500:
            return 1
        else:
            return 0

    async def processLoop(self):
        """Main analysis loop for ISO optimization"""
        try:
            while True:
                try:
                    hist, gray_img, frame, frame_count, current_iso_display = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Throttle analysis rate
                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_analysis_time < self.analysis_interval:
                    continue
                self.last_analysis_time = current_time

                # Skip very dark frames
                frame_mean = np.mean(gray_img)
                if frame_mean < 10:
                    continue

                # Calculate histogram statistics
                hist_flat = hist.flatten()
                levels = np.arange(256)
                mean = float(np.sum(hist_flat * levels) / np.sum(hist_flat))
                median = float(np.median(gray_img))

                dark_ratio = np.sum(gray_img < 30) / gray_img.size
                bright_ratio = np.sum(gray_img > 225) / gray_img.size

                self.mean_buffer.append(mean)
                smooth_mean = np.mean(self.mean_buffer)

                current_iso = self.camera.get_iso()

                # Apply ISO adjustment with cooldown
                if frame_count - self.last_iso_change >= self.iso_change_cooldown:
                    new_iso = self.calculate_optimal_iso(smooth_mean, dark_ratio, bright_ratio)

                    if new_iso != current_iso:
                        self.camera.set_iso(new_iso)
                        self.last_iso_change = frame_count
                        self.stable_frames = 0

                        # Save frame on ISO change
                        if not os.path.exists("auto_adjust_frames"):
                            os.makedirs("auto_adjust_frames", exist_ok=True)
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"auto_adjust_frames/iso_change_{timestamp}_{current_iso}to{new_iso}.jpg"
                        cv.imwrite(filename, frame)
                    else:
                        self.stable_frames += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ANALYZER ERROR] {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main asynchronous entry point"""
    histogram_queue = asyncio.Queue(maxsize=3)

    # Initialize camera interface with fallback
    try:
        camera = create_camera_ctrl("10.98.32.1")  # Replace with actual camera IP
        print("[INIT] Connected to network camera")
    except Exception as e:
        print(f"[WARN] Camera connection failed, using stub: {e}")
        camera = CameraStub()

    # Initialize processing components
    aa = AutoAdjust(queue=histogram_queue, camera=camera, save_frames=True, display_frames=True)
    analyzer = HistogramAnalyzer(histogram_queue, camera)

    # Start concurrent tasks
    auto_task = asyncio.create_task(aa.autoadjustLoop())
    analyzer_task = asyncio.create_task(analyzer.processLoop())

    try:
        await asyncio.gather(auto_task, analyzer_task, return_exceptions=True)
    except KeyboardInterrupt:
        aa.InterruptLoop()
        auto_task.cancel()
        analyzer_task.cancel()
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    """Program entry point"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Program error: {e}")