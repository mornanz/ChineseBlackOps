import numpy as np
import cv2 as cv
import asyncio
import os
from collections import deque
from datetime import datetime
from camera_control import create_camera_ctrl


class AutoAdjust:
    def __init__(self, queue: asyncio.Queue, save_frames=False, display_frames=True):
        self.loopState = False
        self.cap = cv.VideoCapture(0, cv.CAP_ANY)
        self.queue = queue
        self.save_frames = save_frames
        self.display_frames = display_frames
        self.frame_count = 0
        self.save_interval = 30
        self.last_display_time = 0
        self.display_interval = 0.2  # Update display every 200ms (5 FPS)
        
        # Create output directory if saving frames
        if save_frames:
            os.makedirs("auto_adjust_frames", exist_ok=True)

    def CaptureFrame(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return None
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame")
            return None
        return frame

    def HistogramFromFrame(self, frame):
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        return hist, img

    def DisplayFrameWithHistogram(self, frame, gray_img, hist, iso, mean, median):
        """Display frame with histogram overlay"""
        try:
            # Resize frame for consistent display
            display_frame = cv.resize(frame, (640, 480))
            
            # Create histogram visualization with same width as frame
            hist_display = np.zeros((200, 640, 3), dtype=np.uint8)
            
            # Properly extract and normalize histogram values
            hist_flat = hist.flatten()
            if np.sum(hist_flat) > 0:
                hist_normalized = cv.normalize(hist_flat, None, 0, 200, cv.NORM_MINMAX)
                
                # Scale histogram to fit display width
                bin_width = 640 // 256
                for i in range(255):
                    x1 = i * bin_width
                    x2 = (i + 1) * bin_width
                    y1 = 200 - int(hist_normalized[i])
                    y2 = 200 - int(hist_normalized[i + 1])
                    cv.line(hist_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Add mean line (red)
            mean_x = int((mean / 255) * 640)
            cv.line(hist_display, (mean_x, 0), (mean_x, 200), (0, 0, 255), 2)
            
            # Add median line (blue)
            median_x = int((median / 255) * 640)
            cv.line(hist_display, (median_x, 0), (median_x, 200), (255, 0, 0), 2)
            
            # Add target line (green)
            target_x = int((100 / 255) * 640)
            cv.line(hist_display, (target_x, 0), (target_x, 200), (0, 255, 0), 1)
            
            # Add info text to frame
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
            
            # Combine frame and histogram (both now 640 pixels wide)
            combined = np.vstack([display_frame, hist_display])
            cv.imshow("Auto Adjust - Frame & Histogram", combined)
            
        except Exception as e:
            print(f"[DISPLAY ERROR] {e}")

    def InterruptLoop(self):
        print("\nInterrupt received — stopping AutoAdjust loop...")
        self.loopState = False

    async def autoadjustLoop(self):
        try:
            self.loopState = True
            print("AutoAdjust loop started. Press Ctrl+C to stop.")
            print("Press 's' to save current frame, 'q' to quit display window")
            
            while self.loopState:
                frame = self.CaptureFrame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                hist, gray_img = self.HistogramFromFrame(frame)
                
                # Put frame data in queue for analyzer (with current ISO for display)
                current_iso = self.camera.get_iso() if hasattr(self, 'camera') else 800
                try:
                    self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                except asyncio.QueueFull:
                    # If queue is full, remove oldest and add newest
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait((hist, gray_img, frame, self.frame_count, current_iso))
                    except:
                        pass
                
                # Handle display - only update at reduced frame rate
                current_time = asyncio.get_event_loop().time()
                if self.display_frames and (current_time - self.last_display_time >= self.display_interval):
                    # Calculate basic stats for display
                    mean = float(np.mean(gray_img))
                    median = float(np.median(gray_img))
                    
                    self.DisplayFrameWithHistogram(frame, gray_img, hist, current_iso, mean, median)
                    self.last_display_time = current_time
                
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.display_frames = False
                    cv.destroyAllWindows()
                    print("[DISPLAY] Display window closed")
                elif key == ord('s'):
                    # Manual save
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"auto_adjust_frames/manual_frame_{timestamp}.jpg"
                    cv.imwrite(filename, frame)
                    print(f"[MANUAL SAVE] {filename}")

                self.frame_count += 1
                await asyncio.sleep(0.05)  # Small sleep to prevent CPU overload

        except asyncio.CancelledError:
            print("AutoAdjust loop cancelled.")
        except Exception as e:
            print(f"[CAPTURE ERROR] {e}")
        finally:
            self.cap.release()
            cv.destroyAllWindows()
            print("Camera released.")


class HistogramAnalyzer:
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

        # Use the FULL ISO list from camera control
        self.iso_list = self.camera.iso_settings
        print(f"[ISO RANGE] Available ISO values: {self.iso_list}")
        print(f"[ISO RANGE] Min: {min(self.iso_list)}, Max: {max(self.iso_list)}")

    def calculate_optimal_iso(self, smooth_mean, dark_ratio, bright_ratio):
        current_iso = self.camera.get_iso()
        
        # Calculate how far we are from target brightness
        deviation = smooth_mean - self.target_mean
        
        # More aggressive normalization for dark scenes
        normalized_deviation = deviation / 60.0  # Changed from 80 to 60 for more sensitivity
        normalized_deviation = max(-1.0, min(1.0, normalized_deviation))
        
        # More aggressive dark bias
        dark_bias = dark_ratio * 0.8  # Increased from 0.5 to 0.8
        bright_bias = -bright_ratio * 0.5
        
        adjustment_factor = normalized_deviation + dark_bias + bright_bias
        
        print(f"[FUZZY] dev={normalized_deviation:.2f}, dark={dark_bias:.2f}, bright={bright_bias:.2f}, total={adjustment_factor:.2f}")
        
        # Find current ISO index
        try:
            current_idx = self.iso_list.index(current_iso)
        except ValueError:
            current_idx = 2  # Default to 800
        
        # More aggressive ISO changes for dark scenes
        if adjustment_factor < -0.4:  # Very dark
            steps = min(8, int(abs(adjustment_factor) * 10))  # More aggressive
            new_idx = min(current_idx + steps, len(self.iso_list) - 1)
            print(f"[VERY DARK] Increasing ISO by {steps} steps to {self.iso_list[new_idx]}")
        elif adjustment_factor < -0.2:  # Dark
            steps = min(4, int(abs(adjustment_factor) * 6))
            new_idx = min(current_idx + steps, len(self.iso_list) - 1)
            print(f"[DARK] Increasing ISO by {steps} steps to {self.iso_list[new_idx]}")
        elif adjustment_factor < -0.1:  # Slightly dark
            new_idx = min(current_idx + 2, len(self.iso_list) - 1)  # Increased from 1 to 2
            print(f"[SLIGHTLY DARK] Increasing ISO by 2 steps to {self.iso_list[new_idx]}")
        elif adjustment_factor > 0.3:  # Too bright
            steps = min(3, int(adjustment_factor * 3))
            new_idx = max(current_idx - steps, 0)
            print(f"[BRIGHT] Decreasing ISO by {steps} steps to {self.iso_list[new_idx]}")
        elif adjustment_factor > 0.15:  # Slightly bright
            new_idx = max(current_idx - 1, 0)
            print(f"[SLIGHTLY BRIGHT] Decreasing ISO by 1 step to {self.iso_list[new_idx]}")
        else:  # Within acceptable range
            new_idx = current_idx
            print(f"[STABLE] Brightness within target range")
        
        return self.iso_list[new_idx]

    async def processLoop(self):
        print("HistogramAnalyzer started, waiting for data...")
        print(f"[TARGET] Aiming for mean brightness: {self.target_mean}")
        
        try:
            while True:
                try:
                    # Get data with timeout
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

                print(f"[HIST] mean={mean:.1f}, med={median:.1f}, smooth={smooth_mean:.1f}, "
                      f"dark={dark_ratio:.3f}, bright={bright_ratio:.3f}")

                current_iso = self.camera.get_iso()
                
                # Check if we can change ISO
                if frame_count - self.last_iso_change >= self.iso_change_cooldown:
                    new_iso = self.calculate_optimal_iso(smooth_mean, dark_ratio, bright_ratio)
                    
                    if new_iso != current_iso:
                        print(f"[ISO] {current_iso} -> {new_iso}")
                        self.camera.set_iso(new_iso)
                        self.last_iso_change = frame_count
                        self.stable_frames = 0
                        
                        # Save frame when ISO changes
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"auto_adjust_frames/iso_change_{timestamp}_{current_iso}to{new_iso}.jpg"
                        cv.imwrite(filename, frame)
                        print(f"[SAVED ISO CHANGE] {filename}")
                    else:
                        self.stable_frames += 1
                        if self.stable_frames % 10 == 0:
                            print(f"[ISO] stable at {current_iso}")

        except asyncio.CancelledError:
            print("Analyzer loop cancelled.")
        except Exception as e:
            print(f"[ANALYZER ERROR] {e}")


async def main():
    histogram_queue = asyncio.Queue(maxsize=3)  # Very small queue since we're throttling

    camera = create_camera_ctrl("10.98.32.1")
    
    initial_iso = camera.get_iso()
    print(f"[INITIAL] Camera ISO: {initial_iso}")
    
    aa = AutoAdjust(queue=histogram_queue, save_frames=True, display_frames=True)
    # Pass camera reference to AutoAdjust for display
    aa.camera = camera
    analyzer = HistogramAnalyzer(histogram_queue, camera)

    auto_task = asyncio.create_task(aa.autoadjustLoop())
    analyzer_task = asyncio.create_task(analyzer.processLoop())

    try:
        # Run both tasks concurrently
        await asyncio.gather(auto_task, analyzer_task, return_exceptions=True)
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down...")
        aa.InterruptLoop()
        auto_task.cancel()
        analyzer_task.cancel()
        
        # Give tasks time to clean up
        await asyncio.sleep(0.5)
        print("Program exited cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")