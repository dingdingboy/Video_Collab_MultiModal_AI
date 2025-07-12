# -*- coding: utf-8 -*-
import os
from queue import Queue
import sys
import time
import math
import hashlib
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk
import cv2
import threading
import asyncio
import mcp_chat

# Import model handlers on demand to improve startup time
ModelHandler = None
LLMProcessor = None

# Initialize win32 modules (only used for desktop capture)
win32gui = win32ui = win32con = win32api = None
try:
    import win32gui
    import win32ui
    import win32con
    import win32api
    HAVE_WIN32 = True
except ImportError:
    print("Warning: pywin32 is not installed. Installing required packages...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
        print("pywin32 installed successfully. Please restart the application.")
    except Exception as e:
        print(f"Error installing pywin32: {e}")
    HAVE_WIN32 = False

# Basic imports needed for GUI
import numpy as np
from PIL import Image
import cv2  # For camera capture
from list_monitor import list_monitors
from video_similarity import analyze_video_similarity


min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

def get_model_subfolders(model_path=".\\models\\vlm"):
    """
    Gets the names of the subfolders in the specified model directory.

    Args:
        model_path (str): The path to the model directory. Defaults to ".\\models\\vlm".

    Returns:
        list: A list of subfolder names in the model directory.
              Returns an empty list if the directory does not exist or has no subfolders.
    """
    if not os.path.exists(model_path):
        print(f"Error: The directory '{model_path}' does not exist.")
        return []

    subfolders = [f.name for f in os.scandir(model_path) if f.is_dir()]
    return subfolders

def get_quant_type(model_path=".\\models\\vlm\\Qwen2.5-VL-3B-Instruct"):
    """
    Gets the names of the quantization type in the specified model directory.

    Args:
        model_path (str): The path to the model directory. Defaults to ".\\models\\vlm\\Qwen2.5-VL-3B-Instruct".

    Returns:
        list: A list of quantization type names in the model directory.
                Returns an empty list if the directory does not exist or has no subfolders.
    """
    if not os.path.exists(model_path):
        print(f"Error: The directory '{model_path}' does not exist.")
        return []

    subfolders = [f.name for f in os.scandir(model_path) if f.is_dir()]
    return subfolders

def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def inference(video_path, prompt, model_handler, max_new_tokens=2048):
    if not model_handler or not hasattr(model_handler, 'model') or not model_handler.model:
        raise RuntimeError("Model not loaded. Please load the model first.")
        
    return model_handler.run_inference(video_path, prompt, max_new_tokens)


class VideoPlayerApp:    
    def __init__(self, root):
        self.root = root
        self.root.title("AI for Video Meeting")
        
        # Set window icon
        icon = tk.PhotoImage(file='video_collaboration.png')
        self.root.iconphoto(True, icon)
        
        # Video variables
        self.cap = None
        self.desktop_cap = None
        self.is_playing = False
        self.frames_for_inference = []
        self.timestamps_for_inference = []
        self.last_frame_time = 0
        
        # Model handlers - initialize as None and load on demand
        self.model_handler = None
        self.llm_processor = None

        # Create UI elements
        self.create_widgets()        # Update timer
        self.update_interval = 33  # approximately 30 FPS
        self.update_id = None
        self.desktop_update_id = None
        self.inference_timer_id = None  # For continuous inference
        self.is_continuous_inference = False  # Flag for continuous inference
        self.inference_timer_id = None  # For continuous inference
        self.is_continuous_inference = False  # Flag for continuous inference
        self.state_capturing_frame = False  # Flag for indicating the state of capturing 
        self.capture_desktop_interval = 0.5 # Interval to capture desktop frame for inferencing
        self.buffer_size_desktop_inference = 4   # Number of capturing desktop frame for inferencing
        self.capture_camera_interval = 1 # Interval to capture camera frame for inferencing
        self.buffer_size_camera_inference = 4  # # Number of capturing desktop frame for inferencing
        self.capture_camera_frame_num = 0 # Number of capturing camera frame for inferencing
        self.capture_desktop_frame_num = 0 # Number of capturing desktop frame for inferencing
        self.isChatOn = False
        self.mcp_user_message_queue = Queue()  # Queue to hold user messages for MCP chat

    def create_widgets(self):
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Main Video/AI UI ---
        self.tab_vlm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vlm, text="Video Analysis")

        # --- Tab 2: Blank ---
        self.tab_llm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_llm, text="LLM Analysis")
        # Make tab headers and titles larger
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", 16, "bold"), padding=[20, 10])
        style.configure("TNotebook", tabposition='n')

        # --- Place all your existing UI code inside self.tab1 instead of self.root ---
        # Replace all "self.root" with "self.tab1" in widget parents below

        # Create main frame to hold both video displays
        main_frame = ttk.Frame(self.tab_vlm)
        main_frame.pack(pady=10)
        
        # Create frames for each feed and its results
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=10)
          # Camera video canvas, controls and results
        ttk.Label(left_frame, text="Camera Feed").pack()
        self.camera_canvas = tk.Canvas(left_frame, width=960, height=540)
        self.camera_canvas.pack(pady=(0, 5))
        
        # Camera controls in left frame
        camera_controls_frame = ttk.Frame(left_frame)
        camera_controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(camera_controls_frame, text="Camera:").pack(side=tk.LEFT)
        self.cameras = get_available_cameras()
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(camera_controls_frame, textvariable=self.camera_var, state="readonly")
        if self.cameras:
            self.camera_dropdown['values'] = [name for _, name in self.cameras]
            self.camera_var.set(self.cameras[0][1])  # Set default to first camera
        self.camera_dropdown.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(camera_controls_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_controls_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=2)
        
        camera_results_frame = ttk.LabelFrame(left_frame, text="Camera Inference Results")
        camera_results_frame.pack(fill=tk.X, pady=5)
        
        # Add a scrollbar
        self.camera_result_scrollbar = ttk.Scrollbar(camera_results_frame)
        self.camera_result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.camera_result_text = tk.Text(camera_results_frame, height=15, wrap=tk.WORD, yscrollcommand=self.camera_result_scrollbar.set)
        self.camera_result_text.pack(pady=5, padx=5, fill=tk.X, expand=True)

        # Configure the scrollbar
        self.camera_result_scrollbar.config(command=self.camera_result_text.yview)

        # Autoscroll function
        def autoscroll(event):
            self.camera_result_text.see(tk.END)
        
        # Bind the autoscroll function to text changes
        self.camera_result_text.bind("<Insert>", autoscroll)
        self.camera_result_text.bind("<Return>", autoscroll)
          # Desktop video canvas, controls and results
        ttk.Label(right_frame, text="Desktop Feed").pack()
        self.desktop_canvas = tk.Canvas(right_frame, width=960, height=540)
        self.desktop_canvas.pack(pady=(0, 5))
        
        # Desktop controls in right frame
        desktop_controls_frame = ttk.Frame(right_frame)
        desktop_controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(desktop_controls_frame, text="Monitor:").pack(side=tk.LEFT)
        self.monitors = list_monitors()
        self.monitor_var = tk.StringVar()
        self.monitor_dropdown = ttk.Combobox(desktop_controls_frame, textvariable=self.monitor_var, state="readonly")
        if self.monitors:
            self.monitor_dropdown['values'] = [mon['Device'] for i, mon in enumerate(self.monitors, 1)]
            self.monitor_var.set(self.monitors[0]['Device'])  # Set default to first monitor
        self.monitor_dropdown.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(desktop_controls_frame, text="Start Desktop", command=self.start_desktop).pack(side=tk.LEFT, padx=2)
        ttk.Button(desktop_controls_frame, text="Stop Desktop", command=self.stop_desktop).pack(side=tk.LEFT, padx=2)
        
        desktop_results_frame = ttk.LabelFrame(right_frame, text="Desktop Inference Results")
        desktop_results_frame.pack(fill=tk.X, pady=5)

        # Add a scrollbar
        self.desktop_result_scrollbar = ttk.Scrollbar(desktop_results_frame)
        self.desktop_result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.desktop_result_text = tk.Text(desktop_results_frame, height=15, wrap=tk.WORD, yscrollcommand=self.desktop_result_scrollbar.set)
        self.desktop_result_text.pack(pady=5, padx=5, fill=tk.X, expand=True)

        # Configure the scrollbar
        self.desktop_result_scrollbar.config(command=self.desktop_result_text.yview)

        # Autoscroll function
        def autoscroll(event):
            self.desktop_result_text.see(tk.END)
        
        # Bind the autoscroll function to text changes
        self.desktop_result_text.bind("<Insert>", autoscroll)
        self.desktop_result_text.bind("<Return>", autoscroll)
        
        # Inference controls
        inference_frame = ttk.Frame(self.tab_vlm)
        inference_frame.pack(pady=10, fill=tk.X, padx=10)
          # Model selection dropdown
        ttk.Label(inference_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        model_options = get_model_subfolders()
        self.model_var = tk.StringVar(value=model_options[0] if model_options else "")  # Set default value
        self.model_dropdown = ttk.Combobox(inference_frame, textvariable=self.model_var, values=model_options, state="readonly")
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Quantization type dropdown
        ttk.Label(inference_frame, text="Quant Type:").pack(side=tk.LEFT, padx=5)
        self.quant_var = tk.StringVar()
        self.quant_dropdown = ttk.Combobox(inference_frame, textvariable=self.quant_var, state="readonly")
        self.quant_dropdown.pack(side=tk.LEFT, padx=5)

        # Update quant options when model changes
        def update_quant_options(*args):
            selected_model = self.model_var.get()
            if selected_model:
                model_path = os.path.join(".\\models\\vlm", selected_model)
                quant_options = get_quant_type(model_path)
                self.quant_dropdown['values'] = quant_options
                if quant_options:
                    self.quant_var.set(quant_options[0])
                else:
                    self.quant_var.set("")
        
        # Bind the callback to model selection changes
        self.model_var.trace_add('write', update_quant_options)

        # Initialize quant options
        update_quant_options()

        # Device selection dropdown
        ttk.Label(inference_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        device_options = ['GPU', 'CPU']  # Add more devices if needed
        self.device_var = tk.StringVar(value=device_options[0])  # Set default value
        self.device_dropdown = ttk.Combobox(inference_frame, textvariable=self.device_var, values=device_options, state="readonly")
        self.device_dropdown.pack(side=tk.LEFT, padx=5)
        device = self.device_var.get()


        # Load model button
        ttk.Button(inference_frame, text="Load Model", command=self.load_vlm_model_async).pack(side=tk.LEFT, padx=5)
        self.source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(inference_frame, text="Camera Feed", variable=self.source_var, 
                        value="camera").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(inference_frame, text="Desktop Feed", variable=self.source_var, 
                        value="desktop").pack(side=tk.LEFT, padx=5)
        self.status_load_vlm_model = ttk.Label(inference_frame, text="")
        self.status_load_vlm_model.pack(side=tk.LEFT, padx=5)
        
        # Prompt input in a new frame
        prompt_frame = ttk.Frame(self.tab_vlm)
        prompt_frame.pack(pady=5, fill=tk.X, padx=10)
        ttk.Label(prompt_frame, text="Prompt:").pack(side=tk.LEFT)
        self.prompt_text = tk.Text(prompt_frame, height=10, wrap=tk.WORD)
        self.prompt_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Run mode selection
        vlm_infer_frame_holder = ttk.Frame(self.tab_vlm)
        vlm_infer_frame_holder.pack(pady=5, fill=tk.X, padx=10)
        vlm_infer_frame = ttk.Frame(vlm_infer_frame_holder)
        vlm_infer_frame.pack(fill=tk.X, expand=True, padx=5)

        # Center the controls horizontally using an internal frame
        center_frame = ttk.Frame(vlm_infer_frame)
        center_frame.pack(anchor="center")

        ttk.Label(center_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        self.run_mode_var = tk.StringVar(value="Once")
        self.run_mode_dropdown = ttk.Combobox(center_frame, textvariable=self.run_mode_var, 
                              values=["Once", "Every 20 seconds"], 
                              state="readonly", width=15)
        self.run_mode_dropdown.pack(side=tk.LEFT, padx=5)

        ttk.Button(center_frame, text="Inference", command=self.run_inference).pack(side=tk.LEFT, padx=5)
        self.stop_inference_btn = ttk.Button(center_frame, text="Stop", command=self.stop_inference, state=tk.DISABLED)
        self.stop_inference_btn.pack(side=tk.LEFT, padx=5)

        vlm_infer_frame_holder.pack_configure(anchor="center")
        
        # Status label
        self.status_label = ttk.Label(self.tab_vlm, text="")
        self.status_label.pack(pady=5)

        # Model selection controls
        model_control_frame = ttk.LabelFrame(self.tab_llm, text="LLM Model Selection")
        model_control_frame.pack(pady=20,fill=tk.X,padx=10)
        
        
        # LLM Model dropdown
        ttk.Label(model_control_frame, text="LLM Model:").pack(side=tk.LEFT, padx=5)
        self.llm_models = self.get_llm_models()
        self.llm_model_var = tk.StringVar()
        self.llm_model_dropdown = ttk.Combobox(model_control_frame, textvariable=self.llm_model_var, 
                                                values=self.llm_models, state="readonly", width=30)
        if self.llm_models:
            self.llm_model_var.set(self.llm_models[0])
        self.llm_model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Device selection for LLM
        ttk.Label(model_control_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        self.llm_device_var = tk.StringVar(value="GPU")
        self.llm_device_dropdown = ttk.Combobox(model_control_frame, textvariable=self.llm_device_var,
                                                values=["CPU", "GPU"], state="readonly", width=10)
        self.llm_device_dropdown.pack(side=tk.LEFT, padx=5)

        # Load button
        self.llm_load_button = ttk.Button(model_control_frame, text="Load", command=self.load_llm_model)
        self.llm_load_button.pack(side=tk.LEFT, padx=5)

        # Status label for LLM loading
        self.llm_status_load = ttk.Label(model_control_frame, text="")
        self.llm_status_load.pack(side=tk.LEFT, padx=5)

        # LLM controls
        llm_frame = ttk.LabelFrame(self.tab_llm, text="Meeting Summarization")
        llm_frame.pack(pady=5, fill=tk.X, padx=10)

        # LLM Instruction controls
        llm_instruction_frame = ttk.Frame(llm_frame)
        llm_instruction_frame.pack(fill=tk.X, pady=5)
        
        # Instruction
        ttk.Label(llm_instruction_frame, text="Instruction:").pack(side=tk.LEFT, padx=5)
        self.llm_instruction = ttk.Entry(llm_instruction_frame, width=40)
        self.llm_instruction.insert(0, "请给出会议内容的总结，长度控制在100字内")
        self.llm_instruction.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Language selection
        ttk.Label(llm_instruction_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.language_var = tk.StringVar(value="Chinese")
        self.language_dropdown = ttk.Combobox(llm_instruction_frame, textvariable=self.language_var,
                            values=["Chinese", "English"], state="readonly", width=10)
        self.language_dropdown.pack(side=tk.LEFT, padx=5)
        
        # LLM Input controls
        llm_input_frame = ttk.Frame(llm_frame)
        llm_input_frame.pack(fill=tk.X, pady=5,expand=True)

        # Input
        ttk.Label(llm_input_frame, text="Transcript:").pack(side=tk.LEFT, padx=5)
        # Add a scrollbar for the LLM input text box
        self.llm_input_scrollbar = ttk.Scrollbar(llm_input_frame)
        self.llm_input_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.llm_input = tk.Text(llm_input_frame, height=20, wrap=tk.WORD, yscrollcommand=self.llm_input_scrollbar.set)
        self.llm_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.llm_input_scrollbar.config(command=self.llm_input.yview)

        # Inference button
        llm_button_frame = ttk.Frame(llm_frame)
        llm_button_frame.pack(fill=tk.X, pady=5,expand=True)
        self.llm_inference_button = ttk.Button(llm_button_frame, text="Inference", command=self.async_llm_inference)
        self.llm_inference_button.pack(pady=5)
        llm_button_frame.pack_configure(anchor="center")

        # Create a frame to hold the text and scrollbar
        # Create a frame for LLM output, placed next to llm_input_frame
        llm_output_frame = ttk.Frame(llm_frame)
        llm_output_frame.pack(fill=tk.X, pady=5, expand=True)


        # Add a scrollbar
        self.llm_infer_output_scrollbar = ttk.Scrollbar(llm_output_frame)
        self.llm_infer_output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(llm_output_frame, text="Summary:  ").pack(side=tk.LEFT, padx=5)
        # Create the text widget
        self.llm_infer_output_text = tk.Text(llm_output_frame, height=20, wrap=tk.WORD,
                  yscrollcommand=self.llm_infer_output_scrollbar.set)
        self.llm_infer_output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure the scrollbar to scroll the text widget
        self.llm_infer_output_scrollbar.config(command=self.llm_infer_output_text.yview)

        # Autoscroll function
        def autoscroll(event):
            self.llm_infer_output_text.see(tk.END)
        
        # Bind the autoscroll function to text changes
        self.llm_infer_output_text.bind("<Insert>", autoscroll)
        self.llm_infer_output_text.bind("<Return>", autoscroll)

        # LLM Status Label
        self.llm_status_label = ttk.Label(self.tab_llm, text="")
        self.llm_status_label.pack(pady=5)

        # --- MCP Chat Section ---
        mcp_chat_frame = ttk.LabelFrame(self.tab_llm, text="MCP Chat")
        mcp_chat_frame.pack(pady=5, fill=tk.X, padx=10)

        # Chat display box with scrollbar
        # Add a scrollbar for the MCP chat text box
        mcp_chat_session_frame = ttk.Frame(mcp_chat_frame)
        mcp_chat_session_frame.pack(fill=tk.X, pady=5, expand=True)

        ttk.Label(mcp_chat_session_frame, text="Messages: ").pack(side=tk.LEFT, padx=5)
        self.mcp_chat_scrollbar = ttk.Scrollbar(mcp_chat_session_frame)
        self.mcp_chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.mcp_chat_text = tk.Text(mcp_chat_session_frame, height=20, wrap=tk.WORD, state=tk.DISABLED, yscrollcommand=self.mcp_chat_scrollbar.set)
        self.mcp_chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.mcp_chat_scrollbar.config(command=self.mcp_chat_text.yview)

        mcp_user_input_frame = ttk.Frame(mcp_chat_frame)
        mcp_user_input_frame.pack(fill=tk.X, pady=5, expand=True)
        # User input entry
        ttk.Label(mcp_user_input_frame, text="User Input:").pack(side=tk.LEFT, padx=5)
        self.mcp_user_input = ttk.Entry(mcp_user_input_frame, width=60)
        # Bind Enter key to trigger MCP chat send
        self.mcp_user_input.bind("<Return>", lambda event: self.start_mcp_chat())
        self.mcp_user_input.pack(side=tk.LEFT, pady=5, padx=0, fill=tk.X, expand=True)
        self.mcp_chat_button = ttk.Button(mcp_user_input_frame, text="Send", command=self.start_mcp_chat)
        self.mcp_chat_button.pack(side=tk.LEFT,pady=5, padx=5)

        # Start MCP Chat button
        #mcp_user_input_button_frame = ttk.Frame(mcp_chat_frame)
        #mcp_user_input_button_frame.pack(fill=tk.X, pady=5, expand=True)
        #mcp_user_input_button_frame.pack_configure(anchor="center")


    def start_mcp_chat(self):

        user_message = self.mcp_user_input.get().strip()
        if not user_message:
            return
        
        # Check if the LLM model has been loaded or not.
        if self.llm_processor is None or self.llm_processor.model is None:
            self.mcp_chat_text.config(state=tk.NORMAL)
            self.mcp_chat_text.insert(tk.END, f"MCP: Please load the LLM Model first!\n", "mcp")
            self.mcp_chat_text.tag_configure("mcp", foreground="green")
            self.mcp_chat_text.config(state=tk.DISABLED)
            self.mcp_chat_text.see(tk.END)
            return
        
        # Display user message in chat box
        self.mcp_chat_text.config(state=tk.NORMAL)
        self.mcp_chat_text.insert(tk.END, f"User: {user_message}\n", "user")
        self.mcp_chat_text.tag_configure("user", foreground="blue")
        self.mcp_chat_text.config(state=tk.DISABLED)
        self.mcp_chat_text.see(tk.END)
        self.mcp_user_input.delete(0, tk.END)
        self.mcp_user_input.config(state=tk.DISABLED)
        self.mcp_chat_button.config(state=tk.DISABLED)

        def mcp_chat_callback(message):
            # Called by mcp_chat thread with the response
            self.mcp_chat_text.config(state=tk.NORMAL)
            self.mcp_chat_text.insert(tk.END, f"\nMCP: {message}\n", "mcp")
            self.mcp_chat_text.tag_configure("mcp", foreground="green")
            self.mcp_chat_text.config(state=tk.DISABLED)
            self.mcp_chat_text.see(tk.END)
            self.mcp_user_input.config(state=tk.NORMAL)
            self.mcp_chat_button.config(state=tk.NORMAL)

        # Start the mcp_chat thread if not already running
        def chat_thread():
            # This will block and process messages from the queue
            asyncio.run(mcp_chat.mcp_chat_start(self.llm_processor, self.mcp_user_message_queue, mcp_chat_callback))
            self.isChatOn = False
            self.root.after(0, lambda: self.mcp_chat_text.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.mcp_chat_text.delete("1.0", tk.END))

        self.mcp_user_message_queue.put(user_message)

        if not self.isChatOn:
            self.isChatOn = True
            threading.Thread(target=chat_thread, daemon=True).start()

    def load_vlm_model_async(self):
        global ModelHandler
        if ModelHandler is None:
            from model_handler import ModelHandler

        quan_model_path = os.path.join('.\\models\\vlm', self.model_var.get(), self.quant_var.get())
        device = self.device_var.get()
        
        def status_callback(message):
            self.status_load_vlm_model.config(text=message)
        
        # Initialize model handler if not already initialized
        if self.model_handler is None:
            self.model_handler = ModelHandler()
        
        # Start loading thread
        load_thread = self.model_handler.load_model(
            quan_model_path, 
            device, 
            status_callback=status_callback
        )
        load_thread.start()

    def load_llm_model(self):
        """Load the selected LLM model in a separate thread."""
        global LLMProcessor
        if LLMProcessor is None:
            from conf_llm import LLMProcessor

        model_name = self.llm_model_var.get()
        if not model_name:
            self.llm_status_load.config(text="Please select a model")
            return
        
        model_path = os.path.join(".", "models", "llm", model_name)
        device = self.llm_device_var.get()

        def load_thread():
            try:
                start_time = time.time()
                # Initialize processor if not already done
                if self.llm_processor is None:
                    self.llm_processor = LLMProcessor()
                
                if self.llm_processor.load_model(model_path, device):
                    end_time = time.time()
                    loading_time = end_time - start_time
                    self.root.after(0, lambda: self.llm_status_load.config(text=f"LLM Model {model_name} loaded successfully in {loading_time:.2f} seconds"))
                else:
                    self.root.after(0, lambda: self.llm_status_load.config(text="Failed to load LLM model"))
            except Exception as e:
                self.root.after(0, lambda: self.llm_status_load.config(text=f"Error loading model: {str(e)}"))

        # Disable the button while loading
        self.llm_load_button.config(state=tk.DISABLED)
        self.llm_status_load.config(text="Loading LLM model...")
        
        # Start loading in a separate thread
        threading.Thread(target=load_thread, daemon=True).start()
    def get_llm_models(self):
        """
        Gets the names of the subfolders in the specified model directory.

        Args:
            model_path (str): The path to the model directory. Defaults to ".\\models\\llm".

        Returns:
            list: A list of subfolder names in the model directory.
                    Returns an empty list if the directory does not exist or has no subfolders.
        """
        model_path = ".\\models\\llm"
        if not os.path.exists(model_path):
            print(f"Error: The directory '{model_path}' does not exist.")
            return []

        subfolders = [f.name for f in os.scandir(model_path) if f.is_dir()]
        return subfolders
    
    def async_llm_inference(self):
        def llm_inference_thread():
            try:
                self.root.after(0, lambda: self.llm_status_label.config(text="Running LLM inference..."))
                start_time = time.time()

                # Call the predict function in llm_processor
                instruction = self.llm_instruction.get()
                input_text = self.llm_input.get("1.0", tk.END)
                infer_lang = self.language_var.get()
                if self.llm_processor is None:
                    raise RuntimeError("LLM Processor not initialized. Please load the model first.")
                output, input_token_num, out_token_num = self.llm_processor.predict(instruction, input_text, infer_lang)
                
                end_time = time.time()
                inference_time = end_time - start_time

                # Update the result to the text box
                self.root.after(0, lambda: self.llm_infer_output_text.insert(tk.END, f"Input token number: {input_token_num}, output token number: {out_token_num} ", 'red'))
                self.root.after(0, lambda: self.llm_infer_output_text.insert(tk.END, f"Inference time: {inference_time:.2f}\n", 'red'))
                self.llm_infer_output_text.tag_configure('red', foreground='red')
                self.root.after(0, lambda: self.llm_infer_output_text.insert(tk.END, f"{output}\n"))
                self.root.after(0, lambda: self.llm_status_label.config(text=f"LLM Inference complete in {inference_time:.2f} seconds"))
            except Exception as e:
                print(f"Error during LLM inference: {e}")
                self.root.after(0, lambda: self.llm_infer_output_text.insert(tk.END, f"LLM Inference Error: {e}"))
                self.root.after(0, lambda: self.llm_status_label.config(text="LLM Inference failed"))

        # Start a new thread for LLM inference
        threading.Thread(target=llm_inference_thread, daemon=True).start()

    def start_desktop(self):
        if self.desktop_cap is None:
            if not HAVE_WIN32:
                self.status_label.config(text="Error: pywin32 not properly installed. Please restart the application.")
                return None
            
            self.desktop_cap = True  # Just a flag since we're not using cv2.VideoCapture
            self.is_desktop_playing = True
            self.update_desktop_frame()
            self.status_label.config(text="Desktop capture started")

    def stop_desktop(self):
        self.is_desktop_playing = False
        self.desktop_cap = None
        if self.desktop_update_id:
            self.root.after_cancel(self.desktop_update_id)
        self.desktop_canvas.delete("all")     
    def capture_desktop(self):
        if not HAVE_WIN32:
            self.status_label.config(text="Error: pywin32 not properly installed. Please restart the application.")
            return None
        
        # Get selected monitor info
        selected_monitor_name = self.monitor_var.get()
        monitor_info = next((m for m in self.monitors if m['Device'] == selected_monitor_name), None)
        if monitor_info is None:
            return None
            
        #_, _, left, top, width, height = monitor_info
        left = monitor_info['Position'][0]
        top = monitor_info['Position'][1]
        width = int(monitor_info['Resolution'].split('x')[0])
        height = int(monitor_info['Resolution'].split('x')[1])
        
        # Create device context
        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        
        # Create bitmap object
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        
        # Copy screen into the bitmap - only the selected monitor region
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        # Convert the bitmap to an array
        try:
            signedIntsArray = bmp.GetBitmapBits(True)
            img = np.frombuffer(signedIntsArray, dtype='uint8')
            img.shape = (height, width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Error during bitmap conversion: {e}")
            img = None
        
        # Clean up
        memdc.DeleteDC()
        win32gui.DeleteObject(bmp.GetHandle())
        srcdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        
        return img
    
    def update_desktop_frame(self):
        if not self.is_desktop_playing or self.desktop_cap is None:
            return

        frame = self.capture_desktop()
        if frame is not None:
            current_time = time.time()
            
            # Store frame for inference if enough time has passed
            if self.source_var.get() == "desktop" and self.state_capturing_frame == True:
                if self.capture_desktop_frame_num < self.buffer_size_desktop_inference:
                    if (current_time - self.last_frame_time) >= self.capture_desktop_interval: 
                        inference_frame = cv2.resize(frame.copy(), (1280, 720))  # High res copy for inference
                        self.frames_for_inference.append(inference_frame)
                        self.timestamps_for_inference.append(current_time)
                        self.last_frame_time = current_time
                        self.capture_desktop_frame_num += 1
                        self.status_label.config(text=f"Capturing desktop frames")
                else:
                    self.state_capturing_frame = False
                    self.capture_desktop_frame_num = 0
            
            # Resize frame to fit canvas (display version)
            frame = cv2.resize(frame, (960, 540))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(frame_pil)
            
            # Update canvas
            self.desktop_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.desktop_canvas.image = photo  # Keep a reference
        
        # Schedule next update
        self.desktop_update_id = self.root.after(self.update_interval, self.update_desktop_frame)
    def start_camera(self):
        if self.cap is None:
            # Get the selected camera index
            selected_camera_name = self.camera_var.get()
            camera_index = next((index for index, name in self.cameras if name == selected_camera_name), 0)
            
            try:
                # Use DirectShow backend on Windows for faster initialization
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.status_label.config(text=f"Error: Could not open camera {selected_camera_name}")
                    return

                # Set all properties at once using a dictionary
                properties = {
                    cv2.CAP_PROP_FRAME_WIDTH: 960,
                    cv2.CAP_PROP_FRAME_HEIGHT: 540,
                    cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),  # Use MJPG codec
                    cv2.CAP_PROP_BUFFERSIZE: 1  # Minimize frame buffering
                }
                
                # Apply all properties in a batch
                for prop, value in properties.items():
                    self.cap.set(prop, value)

                self.is_playing = True
                self.update_frame()
                self.status_label.config(text=f"Camera {selected_camera_name} started")
            except Exception as e:
                self.status_label.config(text=f"Error initializing camera: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def stop_camera(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.update_id:
            self.root.after_cancel(self.update_id)
        self.status_label.config(text="Camera stopped")
        # Clear the canvas
        self.camera_canvas.delete("all")    
    def update_frame(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            current_time = time.time()
            
            # Store frame for inference if enough time has passed
            if self.source_var.get() == "camera" and self.state_capturing_frame == True:
                if self.capture_camera_frame_num < self.buffer_size_camera_inference:
                    if (current_time - self.last_frame_time) >= self.capture_camera_interval:
                        inference_frame = cv2.resize(frame.copy(), (1280, 720))  # High res copy for inference
                        self.frames_for_inference.append(inference_frame)
                        self.timestamps_for_inference.append(current_time)
                        self.last_frame_time = current_time
                        self.capture_camera_frame_num += 1
                        self.status_label.config(text=f"Capturing camera frames")
                else:
                    self.state_capturing_frame = False
                    self.capture_camera_frame_num = 0
            
            # Convert frame to PhotoImage for display
            frame = cv2.resize(frame, (960, 540))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(frame_pil)
            
            # Update canvas
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.camera_canvas.image = photo  # Keep a reference
        
        # Schedule next update
        self.update_id = self.root.after(self.update_interval, self.update_frame)

    def stop_inference(self):
        if self.inference_timer_id:
            self.root.after_cancel(self.inference_timer_id)
            self.inference_timer_id = None
            self.is_continuous_inference = False
            self.stop_inference_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Continuous inference stopped")

    def run_inference(self):
        source = self.source_var.get()
        result_text = self.camera_result_text if source == "camera" else self.desktop_result_text
        
        # Validate input and capture state
        if source == "camera" and (self.cap is None or not self.cap.isOpened()):
            self.status_label.config(text="Please start camera first! ", foreground='red')
            return
        elif source == "desktop" and self.desktop_cap is None:
            self.status_label.config(text="Please start desktop capture first! ", foreground='red')
            return
        
        prompt = self.prompt_text.get("1.0", tk.END)
        if not prompt or prompt== "\n":
            self.status_label.config(text="Please enter a prompt!", foreground='red')
            return

        # Handler for single inference or continuous inference loop
        def perform_inference():

            # Save frames to video
            temp_dir = '.cache'
            os.makedirs(temp_dir, exist_ok=True)
            temp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()
            frames_file = os.path.join(temp_dir, f'{temp_hash}.mp4')

            # Initilize the buffer for inferencing frame and start the flag of capturing
            self.frames_for_inference = []
            self.timestamps_for_inference = []
            self.state_capturing_frame = True

            # Wait until enough frames are captured
            while self.state_capturing_frame:
                self.root.update()  # Keep the GUI responsive
                time.sleep(0.1)  # Wait a bit before checking again
            
            # Convert frames to video
            height, width, _ = self.frames_for_inference[0].shape
            fourcc = 'mp4v'
            video_writer = cv2.VideoWriter(frames_file, cv2.VideoWriter_fourcc(*fourcc), 2, (width, height))


            for frame in self.frames_for_inference:
                video_writer.write(frame)
            video_writer.release()

            # Run inference in a separate thread
            def inference_thread():
                try:
                    self.status_label.config(text="Running inference...")
                    if source == "desktop":
                        if os.path.exists(os.path.join(temp_dir, 'lasttime.mp4')):
                            if analyze_video_similarity(frames_file, os.path.join(temp_dir, 'lasttime.mp4')) > 10:                                
                                response, inference_time, token_num = inference(frames_file, prompt, self.model_handler)
                            else:
                                response = "Desktop content unchanged, no inference needed.\n"
                                inference_time = 0
                                token_num = 0
                                self.status_label.config(text="Desktop content unchanged")
                        else:
                            response, inference_time, token_num = inference(frames_file, prompt, self.model_handler)
                    else:
                        response, inference_time, token_num = inference(frames_file, prompt, self.model_handler)
                    infer_speed_in_ms = (inference_time * 1000) / token_num if token_num > 0 else 0
                    self.root.after(0, lambda: result_text.insert(tk.END, f"number of token:{token_num}, inference time:{inference_time:.2f}, speed:{infer_speed_in_ms:.3f} ms/token\n", 'red'))
                    self.root.after(0, lambda: result_text.insert(tk.END, response + '\n'))
                    result_text.tag_configure('red', foreground='red')
                    print(f"number of token:{token_num}, inference time:{inference_time:.2f}, speed:{infer_speed_in_ms:.3f} ms/token")
                    print(f'Inference Result -> {prompt}: {response}')
                    self.root.after(0, lambda: self.status_label.config(text="Inference complete"))
                    
                    # Schedule next inference if in continuous mode
                    if self.is_continuous_inference:
                        self.inference_timer_id = self.root.after(20000, perform_inference)  # 20 seconds
                except Exception as e:
                    err_msg = "Inference failed: " + str(e)
                    self.root.after(0, lambda: self.status_label.config(text=err_msg, foreground='red'))
                finally:
                    # Clean up temporary files
                    try:
                        if os.path.exists(os.path.join(temp_dir, 'lasttime.mp4')):
                            os.remove(os.path.join(temp_dir, 'lasttime.mp4'))
                        os.rename(frames_file, os.path.join(temp_dir, 'lasttime.mp4'))
                    except:
                        pass
            
            threading.Thread(target=inference_thread, daemon=True).start()
            def update_timer():
                if self.is_continuous_inference:
                    remaining_time = (20000 - (time.time() - start_time) * 1000) / 1000
                    if remaining_time > 0:
                        self.status_label.config(text=f"Inference complete. Next inference in {remaining_time:.1f} seconds")
                        self.root.after(1000, update_timer)  # Update every 1 second
                    else:
                        self.status_label.config(text="Running inference...")
            start_time = time.time()
            update_timer()

        # Start inference based on selected mode
        if self.run_mode_var.get() == "Every 20 seconds":
            self.is_continuous_inference = True
            self.stop_inference_btn.config(state=tk.NORMAL)

        else:
            self.is_continuous_inference = False
            self.stop_inference_btn.config(state=tk.DISABLED)
            
        perform_inference()
        
    def on_closing(self):
        self.stop_camera()
        self.stop_desktop()
        self.root.destroy()

def get_available_cameras():
    """Get a list of available camera devices and their indices.

    Returns:
        list: A list of tuples containing (index, name) for each available camera.
    """
    if hasattr(get_available_cameras, '_cache'):
        return get_available_cameras._cache

    available_cameras = []
    for i in range(3):  # Check first 3 indices
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if cap.isOpened():
                name = f"Camera {i}"
                available_cameras.append((i, name))
            cap.release()
        except Exception as e:
            print(f"Warning: Error checking camera {i}: {e}")
            continue

    get_available_cameras._cache = available_cameras
    return available_cameras

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()