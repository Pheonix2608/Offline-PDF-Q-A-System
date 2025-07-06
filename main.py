import webview
import threading
import os

def launch_backend():
    os.system("python ui_gradio.py")

# Start Gradio server in background
threading.Thread(target=launch_backend).start()

# Launch native window after short delay
import time
time.sleep(3)

webview.create_window("Offline PDF Assistant", "http://localhost:7860", width=1280, height=800)
webview.start()
