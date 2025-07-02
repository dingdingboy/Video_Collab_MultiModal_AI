import tkinter as tk
import HEC_VC_AI_Sample as hec_ai_simulator

if __name__ == "__main__":

    root = tk.Tk()
    app = hec_ai_simulator.VideoPlayerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()