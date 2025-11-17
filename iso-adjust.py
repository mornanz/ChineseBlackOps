import numpy as np
import cv2 as cv
import asyncio
import os
from collections import deque
from datetime import datetime
from camera_control import create_camera_ctrl

class AutoAdjust:
    """
    Real-time video capture and frame analysis system for automatic ISO adjustment.
    Captures frames from camera, computes histogram statistics, and provides visual feedback.
    """
    
    def __init__(self, queue: asyncio.Queue, save_frames=False, display_frames=True):
        self.loopState = False
        self.cap = cv.VideoCapture(0, cv.CAP_ANY)
        self.queue = queue
        self.save_frames = save_frames
        self.display_frames = display_frames
        self.frame_count = 0
        self.last_display_time = 0
        self.display_interval = 0.2  # 5 FPS display update rate
        
        if save_frames:
            os.makedirs("auto_adjust_frames", exist_ok=True)

    def CaptureFrame(self):
        """Acquire single frame from video capture device with error handling."""
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def HistogramFromFrame(self, frame):
        """Convert frame to grayscale and compute intensity histogram."""
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        return hist, img

    def DisplayFrameWithHistogram(self, frame, gray_img, hist, iso, mean, median):
        """Render frame with overlaid histogram and statistical annotations."""
        try:
            # Normalize display dimensions
            display_frame = cv.resize(frame, (640, 480))
            hist_display = np.zeros((200, 640, 3), dtype=np.uint8)
            
            # Render histogram plot
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
            
            # Annotate statistical markers
            mean_x = int((mean / 255) * 640)
            median_x = int((median / 255) * 640)
            target_x = int((100 / 255) * 640)
            
            cv.line(hist_display, (mean_x, 0), (mean_x, 200), (0, 0, 255), 2)
            cv.line(hist_display, (median_x, 0), (median_x, 200), (255, 0, 0), 2)
            cv.line(hist_display, (target_x, 0), (target_x, 200), (0, 255, 0), 1)
            
            # Overlay text annotations
            cv.putText(display_frame, f"ISO: {iso}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_frame, f"Mean: {mean:.1f}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(display_frame, f"Median: {median:.1f}", (10, 90), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(display_frame, f"Target: 100", (10, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(display_frame, f"Frame: {self.frame_count}", (10, 150),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            combined = np.vstack([display_frame, hist_display])
            cv.imshow("Auto Adjust - Frame & Histogram", combined)
            
        except Exception as e:
            print(f"[DISPLAY ERROR] {e}")

    def InterruptLoop(self):
        """Signal handler for graceful termination."""
        self.loopState = False

    async def autoadjustLoop(self):
        """Main capture loop: acquires frames, computes metrics, and manages display."""
        try:
            self.loopState = True
            
            while self.loopState:
                frame = self.CaptureFrame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                hist, gray_img = self.HistogramFromFrame(frame)
                current_iso = self.camera.get_iso() if hasattr(self, 'camera') else 800
                
                # Queue management with LRU eviction policy
                try:
                    self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                except asyncio.QueueFull:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                    except:
                        pass
                
                # Throttled display update
                current_time = asyncio.get_event_loop().time()
                if self.display_frames and (current_time - self.last_display_time >= self.display_interval):
                    mean = float(np.mean(gray_img))
                    median = float(np.median(gray_img))
                    self.DisplayFrameWithHistogram(frame, gray_img, hist, current_iso, mean, median)
                    self.last_display_time = current_time
                
                # UI event handling
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.display_frames = False
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


class HistogramAnalyzer:
    """
    ISO optimization engine using histogram analysis and brightness stabilization.
    Implements mean-based ISO selection with temporal smoothing and change cooldown.
    """
    
    def __init__(self, queue: asyncio.Queue, camera):
        self.queue = queue
        self.camera = camera
        self.mean_buffer = deque(maxlen=8)
        self.last_iso_change = 0
        self.iso_change_cooldown = 4
        self.target_mean = 100
        self.stable_frames = 0
        self.last_analysis_time = 0
        self.analysis_interval = 0.5
        self.iso_list = self.camera.iso_settings

    def calculate_optimal_iso(self, smooth_mean, dark_ratio, bright_ratio):
        """
        ISO selection with hysteresis to prevent oscillation.
        Maintains current ISO zone unless brightness changes significantly.
        """
        current_iso = self.camera.get_iso()
        
        try:
            current_idx = self.iso_list.index(current_iso)
        except ValueError:
            current_idx = 2

        # Map current ISO to brightness zone for hysteresis
        current_zone = self._get_iso_zone(current_iso)
        
        # Brightness zones with hysteresis thresholds
        if smooth_mean < 20:  # Extreme low-light
            target_zone = 6
            hysteresis_threshold = 25
        elif smooth_mean < 45:  # Severe low-light  
            target_zone = 5
            hysteresis_threshold = 40 if current_zone >= 5 else 25
        elif smooth_mean < 80:  # Moderate low-light
            target_zone = 4
            hysteresis_threshold = 75 if current_zone >= 4 else 40
        elif smooth_mean < 105:  # Slight underexposure
            target_zone = 3
            hysteresis_threshold = 100 if current_zone >= 3 else 75
        elif smooth_mean < 130:  # Target exposure
            target_zone = 2
            hysteresis_threshold = 125 if current_zone >= 2 else 100
        elif smooth_mean < 160:  # Slight overexposure
            target_zone = 1
            hysteresis_threshold = 150 if current_zone >= 1 else 125
        else:  # Severe overexposure
            target_zone = 0
            hysteresis_threshold = 160

        # Apply hysteresis: only change if beyond threshold
        if current_zone != target_zone:
            if (target_zone > current_zone and smooth_mean < hysteresis_threshold) or \
            (target_zone < current_zone and smooth_mean > hysteresis_threshold):
                target_zone = current_zone  # Stay in current zone

        # Map zone to ISO index
        zone_to_iso = {
            6: len(self.iso_list) - 1,      # 102400
            5: len(self.iso_list) - 4,      # ~40000-64000
            4: len(self.iso_list) // 2,     # ~12800-16000  
            3: len(self.iso_list) // 4,     # ~3200-5000 (2000-3200)
            2: 4,                           # 1600 (zamiast 1000-1250) - WIĘKSZY SKOK
            1: 1,                           # 640
            0: 0                            # 500
        }
        
        new_idx = zone_to_iso[target_zone]
        
        if new_idx != current_idx:
            print(f"[HYSTERESIS] mean={smooth_mean:.1f}, zone {current_zone}->{target_zone}, ISO {self.iso_list[current_idx]}->{self.iso_list[new_idx]}")
        else:
            print(f"[HYSTERESIS] mean={smooth_mean:.1f}, staying zone {current_zone}, ISO {self.iso_list[current_idx]}")

        return self.iso_list[new_idx]

    def _get_iso_zone(self, iso):
        """Map ISO value to zone for hysteresis calculation."""
        if iso >= 40000:
            return 6
        elif iso >= 16000:
            return 5  
        elif iso >= 3200:
            return 4
        elif iso >= 1600:  # ZMIENIONE z 1000 na 1600
            return 3
        elif iso >= 640:
            return 2
        elif iso >= 500:
            return 1
        else:
            return 0

    async def processLoop(self):
        """Analysis loop: processes queued frames and adjusts ISO based on brightness metrics."""
        try:
            while True:
                try:
                    hist, gray_img, frame, frame_count, current_iso_display = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Analysis rate limiting
                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_analysis_time < self.analysis_interval:
                    continue
                self.last_analysis_time = current_time

                # Skip invalid frames
                frame_mean = np.mean(gray_img)
                if frame_mean < 10:
                    continue

                # Compute histogram statistics
                hist_flat = hist.flatten()
                levels = np.arange(256)
                mean = float(np.sum(hist_flat * levels) / np.sum(hist_flat))
                median = float(np.median(gray_img))

                dark_ratio = np.sum(gray_img < 30) / gray_img.size
                bright_ratio = np.sum(gray_img > 225) / gray_img.size

                self.mean_buffer.append(mean)
                smooth_mean = np.mean(self.mean_buffer)

                current_iso = self.camera.get_iso()
                
                # ISO adjustment decision with cooldown enforcement
                if frame_count - self.last_iso_change >= self.iso_change_cooldown:
                    new_iso = self.calculate_optimal_iso(smooth_mean, dark_ratio, bright_ratio)
                    
                    if new_iso != current_iso:
                        self.camera.set_iso(new_iso)
                        self.last_iso_change = frame_count
                        self.stable_frames = 0
                        
                        # Document ISO transition
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"auto_adjust_frames/iso_change_{timestamp}_{current_iso}to{new_iso}.jpg"
                        cv.imwrite(filename, frame)
                    else:
                        self.stable_frames += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ANALYZER ERROR] {e}")


async def main():
    """Main application entry point coordinating capture and analysis tasks."""
    histogram_queue = asyncio.Queue(maxsize=3)
    camera = create_camera_ctrl("10.98.32.1")
    
    aa = AutoAdjust(queue=histogram_queue, save_frames=True, display_frames=True)
    aa.camera = camera
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
