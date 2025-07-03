import tkinter as tk
import AI4VideoCollab as aisample

if __name__ == "__main__":

    root = tk.Tk()
    app = aisample.VideoPlayerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()