import os
import time
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np

############################################
# Configuration
############################################

output_dir = "F5-TTS/data/recordings"
os.makedirs(output_dir, exist_ok=True)

# Silence detection and chunk length parameters
min_length = 4  # Minimum chunk length in seconds
max_length = 10  # Maximum chunk length in seconds

# Convert to frame counts
min_frames = int(48000 * min_length)
max_frames = int(48000 * max_length)

# Silence detection parameters
rms_threshold = 10 ** (-40.0 / 20.0)  # Convert -40 dB threshold to linear scale
silence_frames = 0  # Counter for silence duration
max_silence_frames = int(48000 * 2 / 512)  # Max silence allowed (2 seconds at 512-hop length)

# Global variables
current_chunk = []  # Dynamic buffer for the current audio chunk
is_recording = False
stream = None
waiting_for_pause = False  # Flag to indicate we're waiting for a pause after max_length

############################################
# Audio Processing
############################################

def audio_callback(indata, frames, time, status):
    """
    Callback function to handle audio input in real-time.
    Dynamically detects pauses and enforces min/max chunk lengths.
    """
    global current_chunk, silence_frames, waiting_for_pause
    if status:
        print(status)

    # Append audio data to the current chunk
    indata_flat = indata.flatten()
    current_chunk.extend(indata_flat)

    # Calculate RMS and detect silence
    rms = np.sqrt(np.mean(indata_flat**2))
    if rms < rms_threshold:
        silence_frames += 1
    else:
        silence_frames = 0
        waiting_for_pause = False  # Reset waiting if we hear audio again

    # Save the chunk if silence is detected and it meets the length requirements
    if silence_frames >= max_silence_frames and len(current_chunk) >= min_frames:
        save_current_chunk()
        silence_frames = 0  # Reset silence counter
        waiting_for_pause = False  # Reset waiting flag

    # Handle max_length condition
    if len(current_chunk) >= max_frames:
        waiting_for_pause = True  # Mark as waiting for a pause

    # Save if we're waiting for a pause and silence is detected
    if waiting_for_pause and silence_frames >= max_silence_frames:
        save_current_chunk()
        silence_frames = 0
        waiting_for_pause = False  # Reset waiting flag

def save_current_chunk():
    """
    Save the current chunk as a .wav file and reset the buffer.
    """
    global current_chunk
    if current_chunk and len(current_chunk) >= min_frames:
        audio_data = np.array(current_chunk)

        # Normalize the audio chunk
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"chunk_{timestamp}.wav")
        sf.write(output_file, audio_data, samplerate=48000)
        print(f"Saved: {output_file}")

    # Clear the current chunk
    current_chunk = []

def start_recording():
    """
    Start recording audio with silence-based dynamic chunking.
    """
    global is_recording, stream
    if is_recording:
        messagebox.showwarning("Warning", "Recording is already in progress.")
        return

    is_recording = True
    try:
        stream = sd.InputStream(samplerate=48000, channels=1, callback=audio_callback)
        stream.start()
        print("Recording started...")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recording: {e}")

def stop_recording():
    """
    Stop the audio recording and save any remaining audio in the buffer.
    """
    global is_recording, stream, current_chunk
    if not is_recording:
        messagebox.showwarning("Warning", "No recording is in progress.")
        return

    is_recording = False
    try:
        # Stop and close the stream
        stream.stop()
        stream.close()
        print("Recording stopped.")

        # Save any remaining audio in the current chunk
        if current_chunk:
            save_current_chunk()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to stop recording: {e}")

############################################
# Tkinter GUI
############################################

def create_gui():
    """
    Create a GUI for controlling the audio recorder.
    """
    root = tk.Tk()
    root.title("Audio Recorder")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    start_button = tk.Button(frame, text="Start Recording", command=start_recording, width=20)
    start_button.grid(row=0, column=0, padx=5, pady=5)

    stop_button = tk.Button(frame, text="Stop Recording", command=stop_recording, width=20)
    stop_button.grid(row=0, column=1, padx=5, pady=5)

    exit_button = tk.Button(frame, text="Exit", command=root.quit, width=20)
    exit_button.grid(row=1, column=0, columnspan=2, pady=10)

    root.mainloop()

############################################
# Main Execution
############################################

if __name__ == "__main__":
    create_gui()
