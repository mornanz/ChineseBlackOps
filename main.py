"""
Z-CAM Emotion Detection System - FIXED PEOPLE DETECTION for Python 3.12
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
import sys
import ctypes

if sys.platform == "win32":
    python_dir = os.path.dirname(sys.executable)
    os.environ['PATH'] = python_dir + os.pathsep + os.environ.get('PATH', '')
    
    lib_bin = os.path.join(python_dir, "Library", "bin")
    if os.path.exists(lib_bin):
        os.environ['PATH'] = lib_bin + os.pathsep + os.environ.get('PATH', '')
    
    scripts_dir = os.path.join(python_dir, "Scripts")
    if os.path.exists(scripts_dir):
        os.environ['PATH'] = scripts_dir + os.pathsep + os.environ.get('PATH', '')
    
    print(f"[DEBUG] Python dir: {python_dir}")
    print(f"[DEBUG] PATH updated for DLLs")

print("="*60)
print("Z-CAM Emotion Detection System - FIXED PEOPLE DETECTION")
print("="*60)

try:
    import torch
    print(f"[MAIN] PyTorch version: {torch.__version__}")
    print(f"[MAIN] CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"[MAIN WARNING] PyTorch error: {e}")
    print("[MAIN] Trying to continue without GPU support...")

try:
    import numpy as np
    print(f"[MAIN] NumPy version: {np.__version__}")
except ImportError as e:
    print("[MAIN ERROR] NumPy is not installed!")
    print("[MAIN ERROR] Please install NumPy with: pip install numpy")
    sys.exit(1)

import asyncio
import threading
from PyQt5.QtWidgets import QApplication

print("="*60)
print("Z-CAM Emotion Detection System - FIXED PEOPLE DETECTION")
print("="*60)

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from user_interface import CameraUI, UISignals
    print("[MAIN] UI imported")
except ImportError as e:
    print(f"[MAIN ERROR] Cannot import UI: {e}")
    sys.exit(1)

try:
    from iso_adjust_yolo import AutoAdjust, HistogramAnalyzer, CameraStub
    print("[MAIN] AutoAdjust imported")
except ImportError as e:
    print(f"[MAIN ERROR] Cannot import iso_adjust_yolo: {e}")
    sys.exit(1)

# ============================================================================
# FRAME BRIDGE
# ============================================================================

class FrameBridge:
    def __init__(self):
        self.current_frame = None
        self.current_people = []
        self.ui_callback = None
        self.frame_callback = None
    
    def set_ui_callback(self, callback):
        self.ui_callback = callback
    
    def set_frame_callback(self, callback):
        self.frame_callback = callback
    
    def update_frame(self, frame, people):
        self.current_frame = frame.copy()
        self.current_people = people.copy()
        
        if len(people) > 0:
            print(f"[BRIDGE] Got {len(people)} people")
        
        if self.ui_callback:
            self.ui_callback(people)
        
        if self.frame_callback:
            self.frame_callback(frame)
    
    def get_latest_frame(self):
        return self.current_frame

# ============================================================================
# GLOBAL VARIABLE FOR UI ACCESS
# ============================================================================

global_ui = None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def run_combined_app():
    """Run UI and AutoAdjust together"""
    global global_ui
    
    app = QApplication(sys.argv)
    
    bridge = FrameBridge()
    print("[MAIN] Bridge created")
    
    global_ui = CameraUI(bridge)
    print("[MAIN] UI created and stored globally")
    
    global_ui.showMaximized()
    print("[MAIN] UI shown")
    
    def start_auto_adjust():
        print("[AUTOADJUST] Starting...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(start_async_auto_adjust(bridge, global_ui))
        except Exception as e:
            print(f"[AUTOADJUST ERROR] {e}")
        finally:
            loop.close()
    
    adjust_thread = threading.Thread(target=start_auto_adjust, daemon=True)
    adjust_thread.start()
    print("[MAIN] AutoAdjust thread started")
    
    print("[MAIN] Starting PyQt...")
    sys.exit(app.exec_())

async def start_async_auto_adjust(bridge, ui):
    print("[ASYNC] Starting async AutoAdjust...")
    
    histogram_queue = asyncio.Queue(maxsize=3)
    
    camera = CameraStub()
    print("[ASYNC] Camera initialized")
    
    aa = AutoAdjust(
        queue=histogram_queue, 
        camera=camera, 
        frame_bridge=bridge,
        save_frames=False,
        display_frames=False
    )
    print("[ASYNC] AutoAdjust created")
    
    analyzer = HistogramAnalyzer(histogram_queue, camera)
    print("[ASYNC] Analyzer created")
    
    def update_people_callback(people):
        if len(people) > 0:
            print(f"[BRIDGE CALLBACK] Sending {len(people)} people to UI signal")
            if ui and hasattr(ui, 'signals'):
                ui.signals.people_updated.emit(people)
        else:
            print("[BRIDGE CALLBACK] No people detected, sending empty list")
            if ui and hasattr(ui, 'signals'):
                ui.signals.people_updated.emit([])
    
    bridge.set_ui_callback(update_people_callback)
    print("[ASYNC] Callback set with signal emission")
    
    auto_task = asyncio.create_task(aa.autoadjustLoop())
    analyzer_task = asyncio.create_task(analyzer.processLoop())
    print("[ASYNC] Both tasks started")
    
    try:
        await asyncio.gather(auto_task, analyzer_task)
    except asyncio.CancelledError:
        print("[ASYNC] Cancelled")
    except Exception as e:
        print(f"[ASYNC ERROR] {e}")
    finally:
        aa.InterruptLoop()
        print("[ASYNC] Stopped")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n[SYSTEM] Starting...")
    print("[SYSTEM] WARNING: You will see TWO windows:")
    print("  1. PyQt UI (main window)")
    print("  2. OpenCV window (debug - press 'q' to close)")
    print("  3. Check OpenCV window for emotion detection boxes")
    print("-" * 60)
    
    try:
        run_combined_app()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by user")
    except Exception as e:
        print(f"\n[SYSTEM ERROR] {e}")
        import traceback
        traceback.print_exc()