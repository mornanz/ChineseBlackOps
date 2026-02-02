"""
Adaptive ISO Control with Facial Emotion Detection and Histogram Analysis
"""

import numpy as np
import cv2 as cv
import asyncio
import os
from collections import deque
from datetime import datetime
from ultralytics import YOLO
import torch
from camera_control import create_camera_ctrl


# ============================================================================
# CAMERA STUB FOR TESTING
# ============================================================================

class CameraStub:
    """Software-only camera stub for offline testing and development"""
    
    def __init__(self):
        """Initialize with predefined ISO settings"""
        self.iso_settings = [500, 640, 1600, 3200, 6400, 12800, 25600, 51200]
        self._iso = 1600

    def get_iso(self):
        """Return current simulated ISO value"""
        return self._iso

    def set_iso(self, value):
        """Set ISO value if within allowed settings"""
        if value not in self.iso_settings:
            print(f"[CAMERA] Requested ISO {value} not in iso_settings, keeping {self._iso}")
            return
        print(f"[CAMERA] Changing ISO {self._iso} -> {value}")
        self._iso = value


# ============================================================================
# FRAME CAPTURE AND EMOTION DETECTION
# ============================================================================

class AutoAdjust:
    """Real-time frame capture with histogram analysis and emotion detection - UPDATED"""
    
    def __init__(self, queue: asyncio.Queue, camera, frame_bridge, save_frames=False, display_frames=True):
        """
        Initialize capture system with face detection and emotion recognition
        """
        self.loopState = False
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.queue = queue
        self.camera = camera
        self.frame_bridge = frame_bridge
        self.save_frames = save_frames
        self.display_frames = display_frames
        self.frame_count = 0
        self.last_display_time = 0
        self.display_interval = 0.2
        
        self.current_frame = None
        self.detected_people = []

        if save_frames:
            os.makedirs("auto_adjust_frames", exist_ok=True)

        print("[AUTOADJUST] Loading YOLO model best.pt...")
        self.yolo = YOLO("best.pt")
        print("[AUTOADJUST] YOLO classes loaded")

        cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print(f"[WARN] Could not load Haar cascade from {cascade_path}")
            self.face_cascade = None

        self.last_faces = []
        self.missed_frames = 0
        self.max_missed_frames = 5

    def CaptureFrame(self):
        """Capture single frame from video source"""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def HistogramFromFrame(self, frame):
        """Convert frame to grayscale and compute intensity histogram"""
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        return hist, img

    def run_emotion_detection(self, frame, gray_img):
        """Detect faces and classify emotions using stabilized face detection"""
        if self.face_cascade is None:
            return frame.copy(), [] 

        result_frame = frame.copy()
        detected_people = []
        
        gray_blur = cv.GaussianBlur(gray_img, (3, 3), 0)

        faces = self.face_cascade.detectMultiScale(
            gray_blur,
            scaleFactor=1.1,
            minNeighbors=9,
            minSize=(60, 60),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0 and len(self.last_faces) > 0 and self.missed_frames < self.max_missed_frames:
            faces = self.last_faces
            self.missed_frames += 1
        else:
            self.last_faces = faces
            if len(faces) > 0:
                self.missed_frames = 0

        person_id = 1
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            if face_img.size == 0:
                continue

            results = self.yolo.predict(
                face_img,
                conf=0.25,
                iou=0.45,
                imgsz=320,
                verbose=False
            )
            result = results[0]

            text_color = (255, 255, 255)
            shadow_color = (0, 0, 0)

            if len(result.boxes) > 0:
                box = result.boxes[0]
                cls_id = int(box.cls[0])
                label = self.yolo.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])

                text = f"Person {person_id}"
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                text_x = x + 10
                text_y = max(y - 10, 30)
                
                cv.putText(
                    result_frame,
                    text,
                    (text_x + 1, text_y + 1),
                    font,
                    font_scale,
                    shadow_color,
                    thickness,
                    cv.LINE_AA
                )
                
                cv.putText(
                    result_frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv.LINE_AA
                )

                person_data = {
                    'id': person_id,
                    'emotion': label,
                    'confidence': conf,
                    'bbox': (x, y, w, h)
                }
                detected_people.append(person_data)
                
            else:
                text = f"Person {person_id}"
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                text_x = x + 10
                text_y = max(y - 10, 30)
                
                cv.putText(
                    result_frame,
                    text,
                    (text_x + 1, text_y + 1),
                    font,
                    font_scale,
                    shadow_color,
                    thickness,
                    cv.LINE_AA
                )
                
                cv.putText(
                    result_frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv.LINE_AA
                )
                
                person_data = {
                    'id': person_id,
                    'emotion': "Unknown",
                    'confidence': 0.0,
                    'bbox': (x, y, w, h)
                }
                detected_people.append(person_data)
            
            person_id += 1

        return result_frame, detected_people

    def DisplayFrameWithHistogram(self, frame, gray_img, hist, iso, mean, median):
        """Display processed frame with histogram overlay and statistics"""
        try:
            display_frame = cv.resize(frame, (640, 480))
            hist_display = np.zeros((200, 640, 3), dtype=np.uint8)

            hist_flat = hist.flatten()
            if np.sum(hist_flat) > 0:
                hist_normalized = cv.normalize(hist_flat, None, 0, 200, cv.NORM_MINMAX)
                hist_normalized = hist_normalized.flatten()
                bin_width = 640 // 256
                for i in range(255):
                    x1 = i * bin_width
                    x2 = (i + 1) * bin_width
                    y1 = 200 - int(hist_normalized[i])
                    y2 = 200 - int(hist_normalized[i + 1])
                    cv.line(hist_display, (x1, y1), (x2, y2), (0, 255, 0), 1)

            mean_x = int((mean / 255) * 640)
            median_x = int((median / 255) * 640)
            target_x = int((100 / 255) * 640)

            cv.line(hist_display, (mean_x, 0), (mean_x, 200), (0, 0, 255), 2)
            cv.line(hist_display, (median_x, 0), (median_x, 200), (255, 0, 0), 2)
            cv.line(hist_display, (target_x, 0), (target_x, 200), (0, 255, 0), 1)

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

            if len(self.detected_people) > 0:
                cv.putText(display_frame, f"People: {len(self.detected_people)}", (10, 180),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            combined = np.vstack([display_frame, hist_display])
            cv.imshow("Auto Adjust + Person Detection", combined)
        except Exception as e:
            print(f"[DISPLAY ERROR] {e}")

    def InterruptLoop(self):
        """Signal to stop the capture loop"""
        self.loopState = False

    async def autoadjustLoop(self):
        """Main asynchronous capture and processing loop - UPDATED"""
        try:
            self.loopState = True

            while self.loopState:
                frame = self.CaptureFrame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                hist, gray_img = self.HistogramFromFrame(frame)

                result_frame, detected_people = self.run_emotion_detection(frame, gray_img)
                
                self.current_frame = result_frame
                self.detected_people = detected_people
                
                if self.frame_bridge:
                    self.frame_bridge.update_frame(result_frame, detected_people)
                
                current_iso = self.camera.get_iso() if self.camera is not None else 800

                try:
                    self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                except asyncio.QueueFull:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                    except Exception:
                        pass

                current_time = asyncio.get_event_loop().time()
                if self.display_frames and (current_time - self.last_display_time >= self.display_interval):
                    mean = float(np.mean(gray_img))
                    median = float(np.median(gray_img))
                    self.DisplayFrameWithHistogram(result_frame, gray_img, hist, current_iso, mean, median)
                    self.last_display_time = current_time

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
    """ISO optimization based on real-time histogram analysis with hysteresis"""
    
    def __init__(self, queue: asyncio.Queue, camera):
        """
        Initialize histogram analyzer with hysteresis control
        """
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
        """Determine optimal ISO based on smoothed histogram mean with hysteresis"""
        current_iso = self.camera.get_iso()

        try:
            current_idx = self.iso_list.index(current_iso)
        except ValueError:
            current_idx = 2

        current_zone = self._get_iso_zone(current_iso)

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

        if current_zone != target_zone:
            if (target_zone > current_zone and smooth_mean < hysteresis_threshold) or \
               (target_zone < current_zone and smooth_mean > hysteresis_threshold):
                target_zone = current_zone

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
        """Main asynchronous analysis loop for ISO optimization"""
        try:
            while True:
                try:
                    hist, gray_img, frame, frame_count, current_iso_display = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_analysis_time < self.analysis_interval:
                    continue
                self.last_analysis_time = current_time

                frame_mean = np.mean(gray_img)
                if frame_mean < 10:
                    continue

                hist_flat = hist.flatten()
                levels = np.arange(256)
                mean = float(np.sum(hist_flat * levels) / np.sum(hist_flat))
                median = float(np.median(gray_img))

                dark_ratio = np.sum(gray_img < 30) / gray_img.size
                bright_ratio = np.sum(gray_img > 225) / gray_img.size

                self.mean_buffer.append(mean)
                smooth_mean = np.mean(self.mean_buffer)

                current_iso = self.camera.get_iso()

                if frame_count - self.last_iso_change >= self.iso_change_cooldown:
                    new_iso = self.calculate_optimal_iso(smooth_mean, dark_ratio, bright_ratio)

                    if new_iso != current_iso:
                        self.camera.set_iso(new_iso)
                        self.last_iso_change = frame_count
                        self.stable_frames = 0

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
    """Main asynchronous entry point with automatic camera detection"""
    histogram_queue = asyncio.Queue(maxsize=3)

    try:
        camera = create_camera_ctrl("10.98.32.1")
        print("[INIT] Connected to network camera")
    except Exception as e:
        print(f"[WARN] Camera connection failed, using CameraStub: {e}")
        camera = CameraStub()

    aa = AutoAdjust(queue=histogram_queue, camera=camera, save_frames=True, display_frames=True)
    analyzer = HistogramAnalyzer(histogram_queue, camera)

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Program error: {e}")