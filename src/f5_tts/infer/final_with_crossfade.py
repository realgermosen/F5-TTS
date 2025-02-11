import os
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav
import tempfile
import threading
import warnings
import queue  # Import queue for managing audio playback
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

############################################
# Configuration
############################################

# Directories
output_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
generated_dir = "F5-TTS/data/generated"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Audio settings
min_length = 3  # Minimum chunk length in seconds
max_length = 10  # Maximum chunk length in seconds
silence_duration = 1  # Silence threshold for pause detection in seconds
rms_threshold_db = -40.0  # Silence threshold in decibels
min_frames = int(48000 * min_length)
max_frames = int(48000 * max_length)
rms_threshold = 10 ** (rms_threshold_db / 20.0)
max_silence_frames = int(48000 * silence_duration / 512)

# Translation model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True)

# F5-TTS
tts_model_choice = "F5-TTS"
# project = "pentazero"
project = "f5-tts_spanish"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence = True
cross_fade_duration = 0.50 #0.15
speed = 1.0
vocoder = load_vocoder()
vocab_file = os.path.join("F5-TTS/data", f"{project}_char/vocab.txt")
tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)

############################################
# Global Variables
############################################

autoplay_enabled = False
current_chunk = []
is_recording = False
stream = None
silence_frames = 0
waiting_for_pause = False
crossfade_duration = 0.15

# Queue for audio playback
playback_queue = queue.Queue()

stop_event = threading.Event()

# Global crossfaded audio
final_wave_data = None
final_sample_rate = 48000  # default, or adjusted as needed

############################################
# Utility / Helper Functions
############################################

def transcribe_audio(filepath: str, language: str = "En") -> str:
    """
    Transcribes the audio file at `filepath` to text using preprocess_ref_audio_text.
    Returns the transcribed text.
    """
    _, ref_text = preprocess_ref_audio_text(filepath, "", language=language)
    return ref_text

def translate_english_to_spanish(english_text: str) -> str:
    """
    Translates the given English text into Spanish using the NLLB model.
    """
    inputs = tokenizer(english_text, return_tensors="pt")
    forced_bos_token_id = (
        tokenizer.lang_code_to_id["spa_Latn"]
        if hasattr(tokenizer, "lang_code_to_id")
        else tokenizer.convert_tokens_to_ids("spa_Latn")
    )
    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=100
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def run_tts_inference(
    tts_api: F5TTS,
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    remove_silence_flag: bool = True,
    speed_val: float = 1.0
) -> tuple[int, np.ndarray]:
    """
    Runs inference on the TTS model, returning (sample_rate, wave_data).
    """
    seed = -1 # -1 for random
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        final_wave, final_sample_rate, _ = tts_api.infer(
            gen_text=gen_text.lower().strip(),
            ref_text=ref_text.lower().strip(),
            ref_file=ref_audio,
            file_wave=f.name,
            speed=speed_val,
            seed=seed,
            remove_silence=remove_silence_flag,
        )
        sf.write(f.name, final_wave, final_sample_rate)
        if remove_silence_flag:
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = sf.read(f.name)
            final_wave = final_wave.astype("float32")
    return final_sample_rate, final_wave

def crossfade_two_chunks(chunk1: np.ndarray, chunk2: np.ndarray, sr: int, duration: float = 0.15) -> np.ndarray:
    """
    Crossfades two audio chunks and returns a single chunk.
    """
    fade_samples = int(duration * sr)
    if len(chunk1) < fade_samples or len(chunk2) < fade_samples:
        return np.concatenate((chunk1, chunk2))

    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    chunk1[-fade_samples:] *= fade_out

    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    chunk2[:fade_samples] *= fade_in

    return np.concatenate((chunk1, chunk2[fade_samples:]))

def update_final_wave(new_chunk: np.ndarray, sr: int, crossfade_sec: float = 0.15) -> None:
    """
    Crossfades new_chunk into the global final_wave_data.
    """
    global final_wave_data, final_sample_rate
    if final_wave_data is None:
        final_wave_data = new_chunk
        final_sample_rate = sr
    else:
        final_wave_data = crossfade_two_chunks(final_wave_data, new_chunk, sr, duration=crossfade_sec)
    final_sample_rate = sr

def save_crossfaded_output() -> None:
    """
    Saves the global crossfaded output to disk.
    """
    if final_wave_data is not None:
        output_wav_path = os.path.join(generated_dir, "crossfaded_full_output.wav")
        sf.write(output_wav_path, final_wave_data, final_sample_rate)
        print(f"Crossfaded output saved to: {output_wav_path}")

############################################
# Classes
############################################

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

############################################
# Crossfade Chunks (Unchanged from original, except name)
############################################

def crossfade_chunks(chunk1, chunk2, sr, crossfade_duration=0.15):
    """
    Apply a crossfade between two audio chunks.

    Args:
        chunk1 (numpy.ndarray): The first audio chunk.
        chunk2 (numpy.ndarray): The second audio chunk.
        sr (int): Sampling rate.
        crossfade_duration (float): Duration of crossfade in seconds.

    Returns:
        numpy.ndarray: A single chunk with a smooth crossfade transition.
    """
    fade_samples = int(crossfade_duration * sr)

    if len(chunk1) < fade_samples or len(chunk2) < fade_samples:
        return np.concatenate((chunk1, chunk2))

    fade_out = np.linspace(1, 0, fade_samples)
    chunk1[-fade_samples:] *= fade_out

    fade_in = np.linspace(0, 1, fade_samples)
    chunk2[:fade_samples] *= fade_in

    return np.concatenate((chunk1, chunk2[fade_samples:]))

def trim_silence(audio, sr, threshold=-40.0, min_silence_duration=0.1):
    """
    Trim silence from the beginning and end of an audio chunk.
    """
    silence_threshold = 10 ** (threshold / 20.0)
    min_silence_samples = int(min_silence_duration * sr)
    non_silent = np.where(np.abs(audio) > silence_threshold)[0]
    if len(non_silent) < 1:
        return audio
    start_idx = max(0, non_silent[0] - min_silence_samples)
    end_idx = min(len(audio), non_silent[-1] + min_silence_samples)
    return audio[start_idx:end_idx]

############################################
# Audio Playback Worker
############################################

def playback_worker():
    """
    Worker thread to play audio files sequentially from the playback queue.
    """
    while True:
        file_to_play = playback_queue.get()  # Wait for an audio file in the queue
        if file_to_play is None:
            break  # Exit the loop if None is sent
        try:
            wave_data, sample_rate = sf.read(file_to_play)
            sd.play(wave_data, samplerate=sample_rate)
            sd.wait()  # Wait for the playback to finish
        except Exception as e:
            print(f"Error playing audio: {e}")
        playback_queue.task_done()

# Start the playback worker in a separate thread
playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

############################################
# Audio Recording and Processing
############################################

def start_recording():
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
    global is_recording, stream, current_chunk
    if not is_recording:
        messagebox.showwarning("Warning", "No recording is in progress.")
        return
    is_recording = False
    try:
        stream.stop()
        stream.close()
        print("Recording stopped.")
        if current_chunk:
            save_and_process_chunk()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to stop recording: {e}")

def audio_callback(indata, frames, time, status):
    global current_chunk, silence_frames, waiting_for_pause
    if status:
        print(status)

    indata_flat = indata.flatten()
    current_chunk.extend(indata_flat)

    rms = np.sqrt(np.mean(indata_flat**2))
    if rms < rms_threshold:
        silence_frames += 1
    else:
        silence_frames = 0
        waiting_for_pause = False

    if silence_frames >= max_silence_frames and len(current_chunk) >= min_frames:
        save_and_process_chunk()
        silence_frames = 0
        waiting_for_pause = False

    if len(current_chunk) >= max_frames:
        waiting_for_pause = True

    if waiting_for_pause and silence_frames >= max_silence_frames:
        save_and_process_chunk()
        silence_frames = 0
        waiting_for_pause = False

def save_and_process_chunk():
    global current_chunk
    if current_chunk and len(current_chunk) >= min_frames:
        audio_data = np.array(current_chunk)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"chunk_{timestamp}.wav")
        sf.write(output_file, audio_data, samplerate=48000)
        print(f"Saved: {output_file}")
        # Replaces old thread call to process_file with the new approach:
        threading.Thread(target=process_file, args=(output_file,)).start()
    current_chunk = []

############################################
# Updated process_file Function
############################################

def process_file(filepath):
    """
    Transcribe, translate, TTS, crossfade with the global wave, 
    queue playback if enabled, then move original chunk.
    """
    global autoplay_enabled
    try:
        # 1) Transcribe
        ref_text = transcribe_audio(filepath, language="En")
        print(f"Transcribed: {ref_text}")

        # 2) Translate
        translated_text = translate_english_to_spanish(ref_text)
        print(f"Translated: {translated_text}")

        # 3) TTS
        sample_rate, wave_data = run_tts_inference(
            tts_api,
            ref_audio=filepath,
            ref_text=ref_text,
            gen_text=translated_text,
            remove_silence_flag=remove_silence,
            speed_val=speed
        )

        wave_data = wave_data.astype(np.float32)

        # 4) Crossfade into global wave
        update_final_wave(wave_data, sample_rate, crossfade_duration)


        # 5) Save new chunk for playback or future reference
        output_wav_path = os.path.join(generated_dir, f"generated_{os.path.basename(filepath)}")
        sf.write(output_wav_path, wave_data, sample_rate)
        print(f"Generated audio saved to: {output_wav_path}")

        # 6) If autoplay is enabled, queue it
        if autoplay_enabled:
            playback_queue.put(output_wav_path)

        # 7) Move processed chunk
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)

        # 8) Optionally save the crossfaded master wave so far
        save_crossfaded_output()

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

############################################
# Process Media
############################################

def process_media(filepath):
    """
    Process a media file (audio or video) in chunks based on pauses using Slicer.
    """
    try:
        # Extract audio for video files
        temp_audio_path = None
        if filepath.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            print(f"Processing video file: {filepath}")
            with VideoFileClip(filepath) as video:
                audio = video.audio
                temp_audio_path = os.path.join(output_dir, "temp_extracted_audio.wav")
                audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")
                print(f"Extracted audio saved to: {temp_audio_path}")
        elif filepath.lower().endswith(('.wav', '.mp3', '.flac', '.aac')):
            print(f"Processing audio file: {filepath}")
            if not filepath.lower().endswith('.wav'):
                temp_audio_path = os.path.join(output_dir, "converted_audio.wav")
                with AudioFileClip(filepath) as audio:
                    audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")
                print(f"Converted audio saved to: {temp_audio_path}")
            else:
                temp_audio_path = filepath

        # Initialize the queue for sliced chunks
        chunk_queue = queue.Queue()

        def processing_worker():
            while True:
                if stop_event.is_set():
                    break
                chunk_data = chunk_queue.get()
                if chunk_data is None or stop_event.is_set():
                    break
                chunk, start_time, end_time, sr = chunk_data
                temp_chunk_path = os.path.join(output_dir, f"chunk_{start_time}_{end_time}.wav")
                sf.write(temp_chunk_path, chunk, sr)
                # Process the chunk
                process_file(temp_chunk_path)
                chunk_queue.task_done()

        # Start the processing worker as a daemon thread
        threading.Thread(target=processing_worker, daemon=True).start()

        # Read audio file and slice it
        waveform, sr = sf.read(temp_audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        slicer = Slicer(
            sr=sr,
            threshold=-40.0,
            min_length=3000,  # ms
            min_interval=300, # ms
            hop_size=20,      # ms
            max_sil_kept=2000 # ms
        )
        sliced_chunks = slicer.slice(waveform)

        # Enqueue sliced chunks
        for chunk_info in sliced_chunks:
            if isinstance(chunk_info, list) and len(chunk_info) == 3:
                chunk, start_time, end_time = chunk_info
                chunk_queue.put((chunk, start_time / sr, end_time / sr, sr))
            else:
                print("Skipping invalid sliced output:", chunk_info)

        # Wait for processing
        chunk_queue.put(None)
        chunk_queue.join()

    except Exception as e:
        print(f"Error processing media file {filepath}: {e}")

############################################
# Tkinter GUI
############################################

def toggle_autoplay():
    global autoplay_enabled
    autoplay_enabled = not autoplay_enabled
    print(f"Autoplay {'enabled' if autoplay_enabled else 'disabled'}.")

def select_media():
    """
    Opens a file dialog to select a media file (audio or video) and processes it.
    """
    filepath = filedialog.askopenfilename(filetypes=[
        ("Media files", "*.mp4 *.mkv *.avi *.mov *.wav *.mp3 *.flac *.aac"),
        ("Audio files", "*.wav *.mp3 *.flac *.aac"),
        ("Video files", "*.mp4 *.mkv *.avi *.mov"),
    ])
    if filepath:
        threading.Thread(target=process_media, args=(filepath,)).start()

def stop_threads():
    """
    Stop all threads and clean up before exiting.
    """
    stop_event.set()
    playback_queue.put(None)
    print("Exiting application.")

def create_gui():
    root = tk.Tk()
    root.title("Real-Time Translator")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    start_button = tk.Button(frame, text="Start Recording", command=start_recording, width=20)
    start_button.grid(row=0, column=0, padx=5, pady=5)

    stop_button = tk.Button(frame, text="Stop Recording", command=stop_recording, width=20)
    stop_button.grid(row=0, column=1, padx=5, pady=5)

    select_media_button = tk.Button(frame, text="Process Media", command=select_media, width=20)
    select_media_button.grid(row=1, column=0, columnspan=2, pady=5)

    autoplay_check = tk.Checkbutton(frame, text="Autoplay", command=toggle_autoplay)
    autoplay_check.grid(row=2, column=0, columnspan=2, pady=5)

    exit_button = tk.Button(
        frame,
        text="Exit",
        command=lambda: [stop_threads(), root.quit(), root.destroy()],
        width=20
    )
    exit_button.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

############################################
# Main Execution
############################################

if __name__ == "__main__":
    create_gui()
    playback_queue.put(None)  # Stop the playback worker
