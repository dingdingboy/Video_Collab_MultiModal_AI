import ctypes
from ctypes import wintypes, Structure, POINTER, byref, WINFUNCTYPE, windll
import sys

# Define necessary structures and constants
class RECT(Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
    ]

class MONITORINFOEXW(Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32)
    ]
    def __init__(self):
        self.cbSize = ctypes.sizeof(self)
        super().__init__()

# Windows API constants
MONITOR_DEFAULTTONULL = 0
MONITOR_DEFAULTTOPRIMARY = 1
MONITOR_DEFAULTTONEAREST = 2

# Function prototypes
MonitorEnumProc = WINFUNCTYPE(
    wintypes.INT, 
    wintypes.HMONITOR, 
    wintypes.HDC, 
    POINTER(RECT), 
    wintypes.LPARAM
)

def list_monitors():
    monitors = []
    
    @MonitorEnumProc
    def callback(hmonitor, hdc, lprect, lparam):
        info = MONITORINFOEXW()
        ctypes.windll.user32.GetMonitorInfoW(hmonitor, byref(info))
        
        # Extract monitor information
        device_name = info.szDevice
        is_primary = bool(info.dwFlags & 1)  # PRIMARY_MONITOR = 1
        
        # Get monitor position and resolution
        width = info.rcMonitor.right - info.rcMonitor.left
        height = info.rcMonitor.bottom - info.rcMonitor.top
        position = (info.rcMonitor.left, info.rcMonitor.top)
        
        monitors.append({
            "Device": device_name,
            "Primary": is_primary,
            "Resolution": f"{width}x{height}",
            "Position": position,
            "Work Area": f"{info.rcWork.right - info.rcWork.left}x{info.rcWork.bottom - info.rcWork.top}",
            "Handle": hmonitor
        })
        return 1  # Continue enumeration
    
    # Enumerate all monitors
    ctypes.windll.user32.EnumDisplayMonitors(
        0, 0, 
        callback, 0
    )
    return monitors

if __name__ == "__main__":
    all_monitors = list_monitors()
    print(f"Found {len(all_monitors)} monitor(s):\n")
    
    for i, mon in enumerate(all_monitors, 1):
        print(f"Monitor {i}:")
        print(f"  Device: {mon['Device']}")
        print(f"  Primary: {'Yes' if mon['Primary'] else 'No'}")
        print(f"  Resolution: {mon['Resolution']}")
        print(f"  Position: ({mon['Position'][0]}, {mon['Position'][1]})")
        print(f"  Work Area: {mon['Work Area']}")
        print("")