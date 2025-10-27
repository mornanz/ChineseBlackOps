import numpy as np
import cv2 as cv
from datetime import datetime
import os
import asyncio
import signal
import requests

class AutoAdjust:
    def __init__(self, queue: asyncio.Queue):
        self.loopState = False
        self.filename = ""
        self.cap = cv.VideoCapture(cv.CAP_DSHOW)
        self.queue = queue

    def CaptureFrame(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame")
            return
        os.makedirs("./Frames", exist_ok=True)
        self.filename = './Frames/captured_frame' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.jpg'
        cv.imwrite(self.filename, frame)
        print("Frame captured:", self.filename)

    def HistogramFromFrame(self):
        img = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Cannot read image:", self.filename)
            return None
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        self.filename = ""
        return hist

    def InterruptLoop(self):
        print("\nInterrupt received — stopping AutoAdjust loop...")
        self.loopState = False

    async def autoadjustLoop(self):
        try:
            self.loopState = True
            print("AutoAdjust loop started. Press Ctrl+C to stop.")
            while self.loopState:
                self.CaptureFrame()
                hist = self.HistogramFromFrame()
                if hist is not None:
                    await self.queue.put(hist)
                    print("Histogram added to queue.")
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            print("AutoAdjust loop cancelled.")
        finally:
            self.cap.release()
            cv.destroyAllWindows()
            print("Camera released.")


class HistogramAnalyzer:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def processLoop(self):
        print("HistogramAnalyzer started, waiting for data...")
        try:
            while True:
                hist = await self.queue.get()
                levels = np.arange(256)
                mean = float(np.sum(hist[:,0] * levels) / np.sum(hist))
                print(f"[Analyzer] New histogram received. Mean intensity: {mean:.2f}")
        except asyncio.CancelledError:
            print("Analyzer loop cancelled.")

class CameraSettingsChanger:
    def __init__(self, camera_ip):
        self.iso_settings_path_set = "http://" + camera_ip + "/ctrl/set?iso="
        self.iso_settings_path_get = "http://" + camera_ip + "/ctrl/get?k=iso"
        self.current_iso = self.GetCurrentIsoSetting()
        self.iso_settings = [
            500,
            640,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3200,
            4000,
            5000,
            6400,
            8000,
            10000,
            12800,
            16000,
            20000,
            25600,
            32000,
            40000,
            51200,
            64000,
            80000,
            102400,
        ]

    def ListIsoSettings(self):
        return self.iso_settings
    
    def GetCurrentIsoSetting(self):
        response = requests.get(url=self.iso_settings_path_get)
        return response.json().get("value")
        
    def ChangeIsoSetting(self, setting):
        if (setting not in self.iso_settings):
            print("ERROR: tried to change setting to non-existant value")
            return

        response = requests.get(url=self.iso_settings_path_set+setting)
        if response.status_code != 200:
            print("ERROR: change of setings GONE WRONG: " + response.text)
        else:
            self.current_iso = setting


async def main():
    histogram_queue = asyncio.Queue()
    aa = AutoAdjust(queue=histogram_queue)
    analyzer = HistogramAnalyzer(histogram_queue)

    auto_task = asyncio.create_task(aa.autoadjustLoop())
    analyzer_task = asyncio.create_task(analyzer.processLoop())

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        aa.InterruptLoop()
        auto_task.cancel()
        analyzer_task.cancel()
        await asyncio.gather(auto_task, analyzer_task, return_exceptions=True)
        print("Program exited cleanly.")


if __name__ == "__main__":
    # asyncio.run(main())
    csc = CameraSettingsChanger(camera_ip="10.98.32.1")