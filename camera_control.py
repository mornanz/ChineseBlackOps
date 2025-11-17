import subprocess
import requests
import platform

class CameraControl:
    """
    Abstract base class defining camera ISO control interface.
    Implementations must provide concrete ISO get/set operations.
    """
    def set_iso(self, value: int):
        raise NotImplementedError

    def get_iso(self) -> int:
        raise NotImplementedError

class NetworkZCamControl(CameraControl):
    """
    HTTP-based ISO control for Z-CAM cameras via REST API.
    Communicates with camera over network interface.
    """
    
    def __init__(self, ip):
        self.ip = ip
        self.current_iso = None
        self.iso_settings = [
            500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200,
            4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000,
            25600, 32000, 40000, 51200, 64000, 80000, 102400
        ]

    def set_iso(self, value: int):
        """Set camera ISO via HTTP PUT request with timeout handling."""
        try:
            requests.get(f"http://{self.ip}/ctrl/set?iso={value}", timeout=1)
            self.current_iso = value
            print(f"[NET] ISO set to {value}")
        except Exception as e:
            print("[NET ERROR] ISO set failed:", e)

    def get_iso(self) -> int:
        """Retrieve current ISO value with cached fallback for reliability."""
        if self.current_iso is not None:
            return self.current_iso

        try:
            r = requests.get(f"http://{self.ip}/ctrl/get?k=iso", timeout=1)
            self.current_iso = int(r.json().get("value"))
        except:
            print("[NET ERROR] ISO get failed, defaulting to 800")
            self.current_iso = 800

        return self.current_iso


class UsbZCamControl(CameraControl):
    """
    USB-based ISO control using zcamctl command-line utility.
    Provides fallback to cached values when hardware communication fails.
    """
    
    def __init__(self):
        self.current_iso = 800
        self.zcamctl_available = self._check_zcamctl()
        self.iso_settings = [
            500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200,
            4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000,
            25600, 32000, 40000, 51200, 64000, 80000, 102400
        ]

    def _check_zcamctl(self):
        """Verify zcamctl utility availability using platform-specific path resolution."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["where", "zcamctl"], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(["which", "zcamctl"], 
                                      capture_output=True, text=True)
            
            return result.returncode == 0
        except:
            return False

    def set_iso(self, value: int):
        """Execute zcamctl command to modify camera ISO setting."""
        if not self.zcamctl_available:
            print(f"[USB WARNING] zcamctl not available - ISO would change to {value}")
            self.current_iso = value
            return

        try:
            result = subprocess.run(
                ["zcamctl", "--set", f"iso={value}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                self.current_iso = value
                print(f"[USB] ISO set to {value}")
            else:
                print(f"[USB ERROR] zcamctl failed: {result.stderr.strip()}")
        except FileNotFoundError:
            print("[USB ERROR] zcamctl command not found")
            self.zcamctl_available = False
        except Exception as e:
            print("[USB ERROR] ISO set failed:", e)

    def get_iso(self) -> int:
        """Query current ISO via zcamctl with graceful degradation to cache."""
        if not self.zcamctl_available:
            return self.current_iso

        try:
            result = subprocess.run(
                ["zcamctl", "--get", "iso"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                self.current_iso = int(result.stdout.strip())
            else:
                print(f"[USB WARNING] zcamctl get failed: {result.stderr.strip()}")
        except FileNotFoundError:
            self.zcamctl_available = False
        except Exception as e:
            print(f"[USB WARNING] ISO query failed: {e}")

        return self.current_iso


class SimulatedZCamControl(CameraControl):
    """
    Software-only implementation for testing and development.
    Mimics camera behavior without hardware dependencies.
    """
    
    def __init__(self):
        self.current_iso = 800
        self.iso_settings = [
            500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200,
            4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000,
            25600, 32000, 40000, 51200, 64000, 80000, 102400
        ]
        print("[SIM] Using simulated ISO control - no actual camera changes")

    def set_iso(self, value: int):
        """Simulate ISO change by updating internal state only."""
        old_iso = self.current_iso
        self.current_iso = value
        print(f"[SIM] ISO changed from {old_iso} to {value} (simulated)")

    def get_iso(self) -> int:
        """Return current simulated ISO value."""
        return self.current_iso


def create_camera_ctrl(ip="10.98.32.1"):
    """
    Factory function implementing automatic camera control mode detection.
    Priority: Network control > USB control > Simulation mode
    """
    # Network mode detection via HTTP endpoint
    try:
        r = requests.get(f"http://{ip}/ctrl/get?k=iso", timeout=0.6)
        if r.status_code == 200:
            print("[INFO] Z-CAM network control")
            return NetworkZCamControl(ip)
    except:
        pass

    # USB mode detection via zcamctl availability
    usb_control = UsbZCamControl()
    if usb_control.zcamctl_available:
        print("[INFO] Z-CAM USB control via zcamctl")
        return usb_control
    else:
        print("[INFO] Using simulated ISO control (zcamctl not available)")
        return SimulatedZCamControl()