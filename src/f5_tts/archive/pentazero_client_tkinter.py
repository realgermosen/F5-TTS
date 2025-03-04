import os
import time
import threading
import tempfile
import queue
import json
import base64
import socket
import struct
import numpy as np
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox

# For audio playback in Tkinter we use sounddevice to play WAV files.
# (Alternatively, you might use an embedded media player.)
# ------
# Configuration for connection to your server (RTX machine)
SERVER_IP = "192.168.0.19"  # <-- Replace with your server's local IP address
SERVER_PORT = 9998

# Directories for temporary storage (make sure they exist)
OUTPUT_DIR = "F5-TTS/data/recordings"
PROCESSED_DIR = "F5-TTS/data/processed"
GENERATED_DIR = "F5-TTS/data/generated"
for d in [OUTPUT_DIR, PROCESSED_DIR, GENERATED_DIR]:
    os.makedirs(d, exist_ok=True)

# Recording settings
SAMPLERATE = 48000
CHANNELS = 1
MIN_RECORD_SECONDS = 3     # Minimum chunk length (in seconds) before sending
MAX_RECORD_SECONDS = 5     # Maximum chunk length (in seconds)
SILENCE_DURATION = 0.1     # seconds to consider as a pause
RMS_THRESHOLD_DB = -40.0
min_frames = int(SAMPLERATE * MIN_RECORD_SECONDS)
max_frames = int(SAMPLERATE * MAX_RECORD_SECONDS)
rms_threshold = 10 ** (RMS_THRESHOLD_DB / 20.0)
max_silence_frames = int(SAMPLERATE * SILENCE_DURATION / 512)

# Global variables for recording and playback
current_chunk = []   # Accumulate samples here (as a Python list)
is_recording = False
silence_frames = 0
playback_queue = queue.Queue()  # Queue of output file names
autoplay = False   # Default autoplay off

# Lock to synchronize access to current_chunk
chunk_lock = threading.Lock()

# -------------------------------
# CLIENT-SIDE Function to send audio to server
# -------------------------------
def send_audio_to_server(input_audio_file, ref_text="", ref_audio_path_override=None):
    """
    Send the input audio (file path) to the server for processing.
    Optionally, send override reference text and audio.
    Returns the output file path (unique) that the server saves.
    """
    # Read input audio file and encode it.
    with open(input_audio_file, "rb") as f:
        input_audio_data = f.read()
    input_audio_b64 = base64.b64encode(input_audio_data).decode("utf-8")
    # Process optional reference audio override.
    if ref_audio_path_override:
        with open(ref_audio_path_override, "rb") as f:
            ref_audio_data = f.read()
        ref_audio_b64 = base64.b64encode(ref_audio_data).decode("utf-8")
    else:
        ref_audio_b64 = ""
    payload = {
        "audio": input_audio_b64,
        "ref_text": ref_text if ref_text is not None else "",
        "ref_audio": input_audio_b64#ref_audio_b64
    }
    payload_str = json.dumps(payload) + "\n"
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_IP, int(SERVER_PORT)))
        s.sendall(payload_str.encode("utf-8"))
        # Receive audio data until end marker.
        audio_data = b""
        while True:
            chunk = s.recv(4096)
            if b"END_OF_AUDIO" in chunk:
                chunk = chunk.replace(b"END_OF_AUDIO", b"")
                audio_data += chunk
                break
            audio_data += chunk
        s.close()
        # Unpack the floats.
        num_floats = len(audio_data) // 4
        audio_floats = struct.unpack(f"{num_floats}f", audio_data)
        sr = 24000  # Server streams at 24000 Hz.
        # Create a unique filename for output.
        output_file = os.path.join(GENERATED_DIR, f"output_{int(time.time()*1000)}.wav")
        sf.write(output_file, np.array(audio_floats), sr)
        print(f"DEBUG: Saved processed audio to {output_file}")
        return output_file
    except Exception as e:
        print("DEBUG: Error in send_audio_to_server:", e)
        return None

# -------------------------------
# Audio callback and processing functions
# -------------------------------

def audio_callback(indata, frames, time_info, status):
    global current_chunk, silence_frames
    if status:
        print("DEBUG: Recording status:", status)
    with chunk_lock:
        current_chunk.extend(indata.flatten())
        rms = np.sqrt(np.mean(indata.flatten()**2))
        if rms < rms_threshold:
            silence_frames += 1
        else:
            silence_frames = 0
    # If we've detected a pause (enough silence frames) and have enough data, process chunk.
    with chunk_lock:
        if silence_frames >= max_silence_frames and len(current_chunk) >= min_frames:
            process_current_chunk()
            silence_frames = 0
        # Also, if the chunk is too long, process it.
        if len(current_chunk) >= max_frames:
            process_current_chunk()
            silence_frames = 0

def process_current_chunk():
    global current_chunk
    with chunk_lock:
        if not current_chunk:
            return
        # Save the accumulated chunk to a temporary file.
        timestamp = int(time.time()*1000)
        temp_path = os.path.join(OUTPUT_DIR, f"client_chunk_{timestamp}.wav")
        audio_array = np.array(current_chunk)
        # Normalize
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array))
        sf.write(temp_path, audio_array, SAMPLERATE)
        print(f"DEBUG: Saved client chunk to {temp_path}")
        current_chunk = []  # Reset the buffer
    # In a new thread, send this file to the server.
    threading.Thread(target=process_and_queue, args=(temp_path,), daemon=True).start()

def process_and_queue(input_audio_file):
    # You can add code here to pass along any reference overrides from the UI.
    # For simplicity, we'll assume no overrides here.
    output_file = send_audio_to_server(input_audio_file)
    if output_file:
        playback_queue.put(output_file)

# -------------------------------
# Playback Thread
# -------------------------------
def playback_worker():
    while True:
        # Wait for a processed file in the queue.
        output_file = playback_queue.get()
        if output_file is None:
            break
        print(f"DEBUG: Playing {output_file}")
        try:
            # Read and play the file.
            data, sr = sf.read(output_file)
            sd.play(data, samplerate=sr)
            sd.wait()  # Wait until playback finishes.
            print(f"DEBUG: Finished playing {output_file}")
        except Exception as e:
            print("DEBUG: Error during playback:", e)
        playback_queue.task_done()

playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

# -------------------------------
# Tkinter UI
# -------------------------------
class TTSClientApp:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time TTS Client")
        
        self.is_recording = False
        self.stream = None

        # Create UI elements.
        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording, width=20)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording, width=20)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.upload_button = tk.Button(master, text="Upload Audio", command=self.upload_audio, width=20)
        self.upload_button.grid(row=1, column=0, columnspan=2, pady=5)

        self.autoplay_var = tk.IntVar(value=0)
        self.autoplay_check = tk.Checkbutton(master, text="Autoplay", variable=self.autoplay_var, command=self.toggle_autoplay)
        self.autoplay_check.grid(row=2, column=0, columnspan=2, pady=5)

        self.status_label = tk.Label(master, text="Status: Idle")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_recording(self):
        if self.is_recording:
            messagebox.showwarning("Warning", "Already recording.")
            return
        try:
            self.stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=audio_callback)
            self.stream.start()
            self.is_recording = True
            self.status_label.config(text="Status: Recording...")
            print("DEBUG: Recording started.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start recording: {e}")

    def stop_recording(self):
        if not self.is_recording:
            messagebox.showwarning("Warning", "Not recording.")
            return
        try:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.status_label.config(text="Status: Recording stopped.")
            print("DEBUG: Recording stopped.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not stop recording: {e}")

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        if file_path:
            self.status_label.config(text="Status: Processing uploaded file...")
            # In a new thread, send the file to the server.
            threading.Thread(target=self.process_upload, args=(file_path,), daemon=True).start()

    def process_upload(self, file_path):
        output_file = send_audio_to_server(file_path)
        if output_file:
            playback_queue.put(output_file)
            self.status_label.config(text="Status: Uploaded file processed and queued.")
        else:
            self.status_label.config(text="Status: Error processing uploaded file.")

    def toggle_autoplay(self):
        global autoplay
        autoplay = bool(self.autoplay_var.get())
        self.status_label.config(text=f"Status: Autoplay {'ON' if autoplay else 'OFF'}")
        print("DEBUG: Autoplay set to", autoplay)

    def on_close(self):
        if self.is_recording and self.stream:
            self.stream.stop()
            self.stream.close()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TTSClientApp(root)
    root.mainloop()
