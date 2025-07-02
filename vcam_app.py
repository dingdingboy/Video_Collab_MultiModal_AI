import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time  # Import time module for FPS calculation

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Application")        # Set initial window size to match 16:9 aspect ratio plus space for controls
        self.root.geometry("1400x850")  # Width for 1280 video plus margins, height for 720 plus controls

        # Set minimum window size
        self.root.minsize(1400, 850)
        
        self.camera_index = tk.IntVar()
        self.camera_index.set(0)  # Default camera
        
        # Define standard resolution
        self.target_width = 1280
        self.target_height = 720

        self.cap = None
        self.is_running = False
        self.camera_names = {}  # Dictionary to store camera names

        self.create_widgets()

    def create_widgets(self):
        # Main frame to hold everything with improved padding and styling
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        control_panel = ttk.Frame(main_frame)
        control_panel.grid(row=0, column=0, rowspan=2, padx=(0, 20), sticky="ns")

        # Camera selection frame with improved styling
        camera_frame = ttk.LabelFrame(control_panel, text="Camera Selection", padding="10 10 10 10")
        camera_frame.pack(fill="x", pady=(0, 10))

        # Get available cameras
        available_cameras = self.get_available_cameras()
        if not available_cameras:
            ttk.Label(camera_frame, text="No cameras found").pack(pady=10)
            return

        for i, camera in enumerate(available_cameras):
            camera_name = self.camera_names.get(camera, f"Camera {camera}")
            radio = ttk.Radiobutton(camera_frame, text=camera_name, variable=self.camera_index, value=camera)
            radio.pack(pady=2)

        # Start/Stop buttons frame
        button_frame = ttk.LabelFrame(control_panel, text="Camera Controls", padding="10 5 10 5")
        button_frame.pack(fill="x")

        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.start_camera, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Camera feed frame with fixed size
        self.feed_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10 10 10 10")
        self.feed_frame.grid(row=0, column=1, sticky="nsew")
        self.feed_frame.configure(width=self.target_width, height=self.target_height)
        self.feed_frame.grid_propagate(False)

        self.image_label = ttk.Label(self.feed_frame)
        self.image_label.pack(expand=True, fill="both")

        # Configure grid weights for proper resizing
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def get_available_cameras(self):
        available_cameras = []
        for i in range(3):  # Check first 3 camera indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            if cap.isOpened():
                available_cameras.append(i)
                # Try to get camera name (not directly supported by OpenCV)
                camera_name = f"Camera {i}"  # Fallback name
                self.camera_names[i] = camera_name
                cap.release()
            else:
                break  # Stop if a camera cannot be opened
        return available_cameras

    def start_camera(self):
        index = self.camera_index.get()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow for Windows

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {index}")
            return
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Start a separate thread to read and display the camera feed
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_camera(self):
        if self.cap:
            self.is_running = False
            self.cap.release()
            self.cap = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.image_label.config(image='')  # Clear the image

    def update_frame(self):
        try:
            while self.is_running:
                start_time = time.time()  # Start time for FPS calculation
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Resize frame to target resolution
                frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Convert the frame to a format Tkinter can display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)

                # Update the image label
                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep a reference!

        except Exception as e:
            print(f"Error in update_frame: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window closing
    root.mainloop()