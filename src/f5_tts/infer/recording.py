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

def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):  # https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/slicer2.py
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)

class Slicer:  # https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/slicer2.py
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 2000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 2000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("The following condition must be satisfied: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("The following condition must be satisfied: max_sil_kept >= hop_size")
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        ####音频+起始时间+终止时间
        if len(sil_tags) == 0:
            return [[waveform, 0, int(total_frames * self.hop_size)]]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]), 0, int(sil_tags[0][0] * self.hop_size)])
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
                        int(sil_tags[i][1] * self.hop_size),
                        int(sil_tags[i + 1][0] * self.hop_size),
                    ]
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[-1][1], total_frames),
                        int(sil_tags[-1][1] * self.hop_size),
                        int(total_frames * self.hop_size),
                    ]
                )
            return chunks

slicer = Slicer(
    sr=24000, # it was 48000
    threshold=-40.0,
    min_length=5000,
    min_interval=1000,
    hop_size=20,
    max_sil_kept=2000)

# Global variables for recording
is_recording = False
audio_data = []
stream = None

############################################
# Audio Processing
############################################

def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(status)
    audio_data.append(indata.copy())

def start_recording():
    global is_recording, stream, audio_data
    if is_recording:
        messagebox.showwarning("Warning", "Recording is already in progress.")
        return

    is_recording = True
    audio_data = []

    try:
        stream = sd.InputStream(samplerate=48000, channels=1, callback=audio_callback)
        stream.start()
        print("Recording started...")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recording: {e}")

def stop_recording():
    global is_recording, stream
    if not is_recording:
        messagebox.showwarning("Warning", "No recording is in progress.")
        return

    is_recording = False
    try:
        stream.stop()
        stream.close()
        print("Recording stopped.")

        # Process and save audio data
        if audio_data:
            audio_buffer = np.concatenate(audio_data, axis=0).flatten()
            save_recordings(audio_buffer)
            audio_data.clear()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to stop recording: {e}")

def save_recordings(audio_buffer):
    try:
        regions = slicer.slice(audio_buffer)
        print(f"Regions returned by slicer: {regions}")  # Debugging output

        for i, region in enumerate(regions):
            sliced_audio, start, end = region  # Adjusted to match the returned structure
            # Normalize the audio slice
            if np.max(np.abs(sliced_audio)) > 0:
                sliced_audio = sliced_audio / np.max(np.abs(sliced_audio))  # Normalize to [-1, 1]

            output_file = os.path.join(output_dir, f"recording_{int(time.time())}_{i}.wav")
            sf.write(output_file, sliced_audio, samplerate=48000)
            print(f"Recording saved to {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save recordings: {e}")

############################################
# Tkinter GUI
############################################

def create_gui():
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

if __name__ == "__main__":
    create_gui()
