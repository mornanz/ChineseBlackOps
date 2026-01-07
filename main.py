"""
Z-CAM Emotion Detection System - FIXED PEOPLE DETECTION for Python 3.12
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
import sys
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
        
        # DEBUG
        if len(people) > 0:
            print(f"[BRIDGE] Got {len(people)} people")
        
        if self.ui_callback:
            self.ui_callback(people)  # Wywołaj callback z danymi osób
        
        if self.frame_callback:
            self.frame_callback(frame)
    
    def get_latest_frame(self):
        return self.current_frame

# ============================================================================
# GLOBAL VARIABLE FOR UI ACCESS
# ============================================================================

# Globalna referencja do UI, aby callback mógł do niej dotrzeć
global_ui = None

# ============================================================================
# MAIN APPLICATION - POPRAWIONY
# ============================================================================

def run_combined_app():
    """Run UI and AutoAdjust together"""
    global global_ui
    
    app = QApplication(sys.argv)
    
    # Create bridge
    bridge = FrameBridge()
    print("[MAIN] Bridge created")
    
    # Create UI and store globally
    global_ui = CameraUI(bridge)
    print("[MAIN] UI created and stored globally")
    
    # Show UI
    global_ui.showMaximized()
    print("[MAIN] UI shown")
    
    # Start AutoAdjust in separate thread
    def start_auto_adjust():
        print("[AUTOADJUST] Starting...")
        
        # Utwórz nową pętlę asyncio dla tego wątku
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(start_async_auto_adjust(bridge, global_ui))
        except Exception as e:
            print(f"[AUTOADJUST ERROR] {e}")
        finally:
            loop.close()
    
    # Start AutoAdjust thread
    adjust_thread = threading.Thread(target=start_auto_adjust, daemon=True)
    adjust_thread.start()
    print("[MAIN] AutoAdjust thread started")
    
    # Run PyQt
    print("[MAIN] Starting PyQt...")
    sys.exit(app.exec_())

async def start_async_auto_adjust(bridge, ui):
    """Asynchroniczna część AutoAdjust z dostępem do UI"""
    print("[ASYNC] Starting async AutoAdjust...")
    
    # Create queue
    histogram_queue = asyncio.Queue(maxsize=3)
    
    # Create camera stub
    camera = CameraStub()
    print("[ASYNC] Camera initialized")
    
    # Create AutoAdjust - POKAŻ OKNO OpenCV dla debugu!
    aa = AutoAdjust(
        queue=histogram_queue, 
        camera=camera, 
        frame_bridge=bridge,
        save_frames=False,
        display_frames=False  # WŁĄCZONE - pokaż okno OpenCV
    )
    print("[ASYNC] AutoAdjust created")
    
    # Create analyzer
    analyzer = HistogramAnalyzer(histogram_queue, camera)
    print("[ASYNC] Analyzer created")
    
    # Ustaw callback dla osób - WYWOŁUJ SIGNAL!
    def update_people_callback(people):
        """Callback dla osób - wysyła dane do UI przez signal"""
        if len(people) > 0:
            print(f"[BRIDGE CALLBACK] Sending {len(people)} people to UI signal")
            # Użyj metody invokeLater dla bezpieczeństwa między wątkami
            if ui and hasattr(ui, 'signals'):
                # Emit signal w głównym wątku Qt
                ui.signals.people_updated.emit(people)
        else:
            print("[BRIDGE CALLBACK] No people detected, sending empty list")
            if ui and hasattr(ui, 'signals'):
                ui.signals.people_updated.emit([])
    
    bridge.set_ui_callback(update_people_callback)
    print("[ASYNC] Callback set with signal emission")
    
    # Start both tasks
    auto_task = asyncio.create_task(aa.autoadjustLoop())
    analyzer_task = asyncio.create_task(analyzer.processLoop())
    print("[ASYNC] Both tasks started")
    
    try:
        # Wait for tasks to complete
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