import os
import time
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav
import tempfile
import threading
import warnings

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
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)

# F5-TTS
tts_model_choice = "F5-TTS"
project = "pentazero_v0" ## "f5-tts_spanish"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence = True
cross_fade_duration = 0.15
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

current_chunk = []
is_recording = False
stream = None
silence_frames = 0
waiting_for_pause = False

############################################
# Audio Recording and Processing
############################################

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
        threading.Thread(target=process_file, args=(output_file,)).start()
    current_chunk = []

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

############################################
# Translation and TTS
############################################

def translate_text(english_text):
    inputs = tokenizer(english_text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id["spa_Latn"] if hasattr(tokenizer, "lang_code_to_id") else tokenizer.convert_tokens_to_ids("spa_Latn")
    translated_tokens = translation_model.generate(
        **inputs, forced_bos_token_id=forced_bos_token_id, max_length=100
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def run_infer(tts_api, ref_audio, ref_text, gen_text):
    seed = -1
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        final_wave, final_sample_rate, _ = tts_api.infer(
            gen_text=gen_text.lower().strip(),
            ref_text=ref_text.lower().strip(),
            ref_file=ref_audio,
            file_wave=f.name,
            speed=speed,
            seed=seed,
            remove_silence=remove_silence,
        )
        sf.write(f.name, final_wave, final_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = sf.read(f.name)
            final_wave = final_wave.astype("float32")
    return final_sample_rate, final_wave

def process_file(filepath):
    try:
        ref_text = preprocess_ref_audio_text(filepath, "", language="En")[1]
        translated_text = translate_text(ref_text)
        print(f"Transcribed: {ref_text}")
        print(f"Translated: {translated_text}")
        sample_rate, wave_data = run_infer(tts_api, filepath, ref_text, translated_text)
        output_wav_path = os.path.join(generated_dir, f"generated_{os.path.basename(filepath)}")
        sf.write(output_wav_path, wave_data, sample_rate)
        print(f"Generated audio saved to: {output_wav_path}")
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

############################################
# Tkinter GUI
############################################

def create_gui():
    root = tk.Tk()
    root.title("Real-Time Translator")

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
