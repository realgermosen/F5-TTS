import os
import time
import tempfile
import threading
import queue
import warnings

import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# F5-TTS imports
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav

# MoviePy imports
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

############################################
# Configuration / Global Variables
############################################

# Directories for saving files
output_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
generated_dir = "F5-TTS/data/generated"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Audio settings
sample_rate = 48000
min_length = 3   # Minimum chunk length in seconds
max_length = 8  # Maximum chunk length in seconds
silence_duration = 1  # Silence threshold for pause detection in seconds
rms_threshold_db = -40.0  # Silence threshold in decibels
min_frames = int(sample_rate * min_length)
max_frames = int(sample_rate * max_length)
rms_threshold = 10 ** (rms_threshold_db / 20.0)
max_silence_frames = int(sample_rate * silence_duration / 512)

# Recording / silence detection state
is_recording = False
stream = None
current_chunk = []
silence_frames = 0
waiting_for_pause = False

# Playback / crossfade
autoplay_enabled = False
playback_queue = queue.Queue()
stop_event = threading.Event()
final_wave_data = None
final_sample_rate = sample_rate
crossfade_duration = 0.15

# Translation model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True)

# F5-TTS
tts_model_choice = "F5-TTS"
project = "pentazero_carlos" ## "f5-tts_spanish"  # or whichever project folder name you use
ckpt_path = f"F5-TTS/ckpts/{project}/model_8200000.pt"
remove_silence_flag = True
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
# Helper Functions
############################################

def _log(logs, message: str):
    """
    Append a message to the logs, limit to last ~30 entries,
    and return both updated logs and the display string.
    """
    logs.append(message)
    # Keep the log from getting too long
    logs = logs[-30:]
    return logs, "\n".join(logs)

def transcribe_audio(filepath: str, language: str = "En") -> str:
    """
    Transcribes the audio file at `filepath` using F5-TTS's helper, returning text.
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

def run_tts_inference(ref_audio: str, ref_text: str, gen_text: str) -> tuple[int, np.ndarray]:
    """
    Runs inference on the TTS model, returning (sample_rate, wave_data).
    `ref_audio` is the chunk we used for reference (tone), 
    `ref_text` is the original text in English,
    `gen_text` is the Spanish text after translation.
    """
    seed = -1  # -1 for random seed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        final_wave, final_sr, _ = tts_api.infer(
            gen_text=gen_text.lower().strip(),
            ref_text=ref_text.lower().strip(),
            ref_file=ref_audio,
            file_wave=f.name,
            speed=speed,
            seed=seed,
            remove_silence=remove_silence_flag,
        )
        sf.write(f.name, final_wave, final_sr)
        if remove_silence_flag:
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = sf.read(f.name)
            final_wave = final_wave.astype("float32")
    return final_sr, final_wave

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

def update_final_wave(new_chunk: np.ndarray, sr: int, crossfade_sec: float = 0.15):
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

def save_crossfaded_output():
    """
    Saves the global crossfaded output to disk.
    """
    if final_wave_data is not None:
        output_wav_path = os.path.join(generated_dir, "crossfaded_full_output.wav")
        sf.write(output_wav_path, final_wave_data, final_sample_rate)
        print(f"Crossfaded output saved to: {output_wav_path}")

############################################
# Audio Callback & Recording
############################################

def audio_callback(indata, frames, time_info, status):
    global current_chunk, silence_frames, waiting_for_pause, is_recording
    if status:
        print(status)

    if not is_recording:
        return

    indata_flat = indata.flatten()
    current_chunk.extend(indata_flat)
    rms = np.sqrt(np.mean(indata_flat**2))

    # Track silence frames
    if rms < rms_threshold:
        silence_frames += 1
    else:
        silence_frames = 0
        waiting_for_pause = False

    # If silence is long enough, and we have enough recorded frames
    if silence_frames >= max_silence_frames and len(current_chunk) >= min_frames:
        save_and_process_chunk()
        silence_frames = 0
        waiting_for_pause = False

    # If we exceed the max chunk length, wait for next silence
    if len(current_chunk) >= max_frames:
        waiting_for_pause = True

    # If we’re waiting for pause & we detect enough silence
    if waiting_for_pause and silence_frames >= max_silence_frames:
        save_and_process_chunk()
        silence_frames = 0
        waiting_for_pause = False

def save_and_process_chunk():
    """
    Saves the audio chunk to disk, then dispatches a thread to process it.
    """
    global current_chunk
    if current_chunk and len(current_chunk) >= min_frames:
        audio_data = np.array(current_chunk)
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"chunk_{timestamp}.wav")
        sf.write(output_file, audio_data, samplerate=sample_rate)
        print(f"Saved: {output_file}")
        # Process in a separate thread
        threading.Thread(target=process_file, args=(output_file,)).start()
    current_chunk = []

############################################
# File Processing
############################################

def process_file(filepath):
    """
    1) Transcribe chunk
    2) Translate text
    3) TTS on Spanish text
    4) Crossfade to global wave
    5) Save chunk
    6) If autoplay, queue for playback
    7) Move the processed chunk
    8) Save crossfaded master
    """
    global autoplay_enabled
    try:
        # We won't use _log inside the thread by default, as it's separate from a user event.
        # Instead, just print or optionally store logs as needed.
        print(f"--- Processing file: {filepath} ---")

        # 1) Transcribe
        print("Step 1/8: Transcribing audio...")
        ref_text = transcribe_audio(filepath, language="En")
        print(f"Finished transcription: {ref_text}")

        # 2) Translate
        print("Step 2/8: Translating English → Spanish...")
        translated_text = translate_english_to_spanish(ref_text)
        print(f"Finished translation: {translated_text}")

        # 3) TTS
        print("Step 3/8: Generating TTS audio...")
        sr, wave_data = run_tts_inference(filepath, ref_text, translated_text)
        print("TTS audio generated")

        # 4) Update global wave
        print("Step 4/8: Crossfading into master track...")
        update_final_wave(wave_data, sr, crossfade_duration)
        print("Crossfade complete")

        # 5) Save TTS chunk
        print("Step 5/8: Saving TTS output to disk...")
        output_wav_path = os.path.join(generated_dir, f"generated_{os.path.basename(filepath)}")
        sf.write(output_wav_path, wave_data, sr)
        print(f"TTS chunk saved: {output_wav_path}")

        # 6) If autoplay is enabled, queue it
        if autoplay_enabled:
            print("Step 6/8: Adding chunk to autoplay queue...")
            playback_queue.put(output_wav_path)

        # 7) Move processed chunk
        print("Step 7/8: Moving original file to 'processed' folder...")
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)
        print(f"Chunk moved to: {processed_path}")

        # 8) Save the crossfaded output
        print("Step 8/8: Saving crossfaded master output...")
        save_crossfaded_output()
        print("All steps completed successfully.\n")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

############################################
# Audio Playback Worker
############################################

def playback_worker():
    """
    Plays audio files from playback_queue in sequence.
    """
    while True:
        if stop_event.is_set():
            break
        file_to_play = playback_queue.get()
        if file_to_play is None or stop_event.is_set():
            break
        try:
            wave_data, sr = sf.read(file_to_play)
            sd.play(wave_data, samplerate=sr)
            sd.wait()  # block until finished
        except Exception as e:
            print(f"Error playing audio: {e}")
        playback_queue.task_done()

playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

############################################
# Slicing Logic (Same from original)
############################################

def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

class Slicer:
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
            raise ValueError("Condition must hold: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("Condition must hold: max_sil_kept >= hop_size")
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
        for i, rms_val in enumerate(rms_list):
            if rms_val < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
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
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
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
# process_media
############################################

def process_media(filepath: str):
    """
    Process an audio or video file in chunks based on silence.
    """
    try:
        temp_audio_path = None

        # If it's a video file, extract audio
        if filepath.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            print(f"Processing video file: {filepath}")
            with VideoFileClip(filepath) as video:
                audio = video.audio
                temp_audio_path = os.path.join(output_dir, "temp_extracted_audio.wav")
                audio.write_audiofile(temp_audio_path, fps=sample_rate, nbytes=2, codec="pcm_s16le")
                print(f"Extracted audio saved to: {temp_audio_path}")
        elif filepath.lower().endswith(('.wav', '.mp3', '.flac', '.aac')):
            print(f"Processing audio file: {filepath}")
            if not filepath.lower().endswith('.wav'):
                temp_audio_path = os.path.join(output_dir, "converted_audio.wav")
                with AudioFileClip(filepath) as audio:
                    audio.write_audiofile(temp_audio_path, fps=sample_rate, nbytes=2, codec="pcm_s16le")
                print(f"Converted audio saved to: {temp_audio_path}")
            else:
                temp_audio_path = filepath
        else:
            return f"Unsupported file format: {filepath}"

        # read audio, slice by silence, and queue chunks for processing
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

        # For each chunk, save and process in a separate thread
        for chunk_info in sliced_chunks:
            if isinstance(chunk_info, list) and len(chunk_info) == 3:
                chunk, start_time, end_time = chunk_info
                temp_chunk_path = os.path.join(output_dir, f"chunk_{start_time}_{end_time}.wav")
                sf.write(temp_chunk_path, chunk, sr)
                threading.Thread(target=process_file, args=(temp_chunk_path,)).start()
            else:
                print("Skipping invalid chunk output:", chunk_info)

        return "Media processing started. Check logs for updates."
    except Exception as e:
        return f"Error processing media file {filepath}: {e}"

############################################
# Gradio App
############################################

def on_start_recording(logs):
    """
    Starts the audio stream for real-time recording.
    """
    global is_recording, stream, current_chunk
    if is_recording:
        return _log(logs, "Recording is already in progress.")
    try:
        is_recording = True
        current_chunk = []
        stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
        stream.start()
        return _log(logs, "Recording started...")
    except Exception as e:
        return _log(logs, f"Failed to start recording: {e}")

def on_stop_recording(logs):
    """
    Stops the audio stream for real-time recording.
    """
    global is_recording, stream
    if not is_recording:
        return _log(logs, "No recording is in progress.")
    try:
        is_recording = False
        stream.stop()
        stream.close()
        save_and_process_chunk()
        return _log(logs, "Recording stopped.")
    except Exception as e:
        return _log(logs, f"Failed to stop recording: {e}")

def on_toggle_autoplay(logs, current_value):
    """
    Toggles autoplay checkbox and returns updated log.
    """
    global autoplay_enabled
    autoplay_enabled = current_value
    return _log(logs, f"Autoplay {'enabled' if autoplay_enabled else 'disabled'}.")

def on_process_media(logs, file_obj):
    """
    Takes the uploaded file from Gradio, processes it in chunks by silence.
    """
    try:
        if file_obj is None:
            return _log(logs, "No file was uploaded.")
        msg = process_media(file_obj.name)
        return _log(logs, msg)
    except Exception as e:
        return _log(logs, f"Error processing media: {e}")

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Real-Time Translator (Gradio Version)")
        gr.Markdown(
            "Records live audio in English, transcribes, translates to Spanish, then TTS in Spanish."
        )

        # State to store logs
        log_state = gr.State([])

        # Button row
        with gr.Row():
            start_btn = gr.Button("Start Recording", variant="primary")
            stop_btn = gr.Button("Stop Recording", variant="stop")
            autoplay_cb = gr.Checkbox(label="Autoplay", value=False)

        # File upload for media
        media_upload = gr.File(
            file_types=["audio", "video"],
            label="Upload an audio or video file to process in chunks"
        )

        # Log viewer
        output_log = gr.Textbox(
            label="Logs",
            lines=10,
            interactive=False
        )

        # Button actions → all go through our log-based functions
        start_btn.click(
            fn=on_start_recording,
            inputs=[log_state],
            outputs=[log_state, output_log]
        )
        stop_btn.click(
            fn=on_stop_recording,
            inputs=[log_state],
            outputs=[log_state, output_log]
        )
        autoplay_cb.change(
            fn=on_toggle_autoplay,
            inputs=[log_state, autoplay_cb],
            outputs=[log_state, output_log]
        )
        media_upload.change(
            fn=on_process_media,
            inputs=[log_state, media_upload],
            outputs=[log_state, output_log]
        )

        # Queue so that long-running tasks don’t block other user interactions
        demo.queue()
    return demo

app = build_interface()

if __name__ == "__main__":
    app.launch(share=True)
