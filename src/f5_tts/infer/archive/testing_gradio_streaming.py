import os
import time
import threading
import tempfile
import warnings
import numpy as np
import soundfile as sf
import gradio as gr

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

############################################
# Configuration and Directories
############################################

output_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
generated_dir = "F5-TTS/data/generated"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Audio settings (in seconds)
min_length = 3      # minimum chunk duration
max_length = 5      # maximum chunk duration
silence_duration = 0.1  
rms_threshold_db = -40.0  
min_frames = int(48000 * min_length)
max_frames = int(48000 * max_length)
rms_threshold = 10 ** (rms_threshold_db / 20.0)
max_silence_frames = int(48000 * silence_duration / 512)

############################################
# Models and TTS Setup
############################################

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True)

tts_model_choice = "F5-TTS"
project = "f5-tts_spanish"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence_flag = False
cross_fade_duration = 0.50  
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
# Global Variables for Processing
############################################

# Buffer for client (Orlando) audio
client_buffer = []  

# Global synthesized audio (merged output)
final_wave_data = None
final_sample_rate = 48000

############################################
# Utility Functions
############################################

def transcribe_audio(filepath: str, language: str = "En") -> str:
    _, ref_text = preprocess_ref_audio_text(filepath, "", language=language)
    return ref_text

def translate_english_to_spanish(english_text: str) -> str:
    inputs = tokenizer(english_text, return_tensors="pt")
    forced_bos_token_id = (tokenizer.lang_code_to_id["spa_Latn"]
                           if hasattr(tokenizer, "lang_code_to_id")
                           else tokenizer.convert_tokens_to_ids("spa_Latn"))
    translated_tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=100
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def clip_audio(input_filepath: str, max_duration_sec: int = 6, output_filepath: str = None) -> str:
    data, sr = sf.read(input_filepath)
    if data.ndim > 1:
        data = data[:, 0]
    max_samples = int(max_duration_sec * sr)
    clipped_data = data[:max_samples] if len(data) > max_samples else data
    if output_filepath is None:
        base = os.path.basename(input_filepath)
        output_filepath = os.path.join(output_dir, f"clipped_{base}")
    sf.write(output_filepath, clipped_data, sr)
    return output_filepath

def run_tts_inference(tts_api, ref_audio: str, ref_text: str, gen_text: str,
                      remove_silence_flag: bool = True, speed_val: float = 1.0):
    seed = -1
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
    fade_samples = int(duration * sr)
    if len(chunk1) < fade_samples or len(chunk2) < fade_samples:
        return np.concatenate((chunk1, chunk2))
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    chunk1[-fade_samples:] *= fade_out
    fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
    chunk2[:fade_samples] *= fade_in
    return np.concatenate((chunk1, chunk2[fade_samples:]))

def update_final_wave(new_chunk: np.ndarray, sr: int, crossfade_sec: float = 0.15) -> None:
    global final_wave_data, final_sample_rate
    if final_wave_data is None:
        final_wave_data = new_chunk
        final_sample_rate = sr
    else:
        final_wave_data = crossfade_two_chunks(final_wave_data, new_chunk, sr, duration=crossfade_sec)
    final_sample_rate = sr

def get_final_audio():
    global final_wave_data, final_sample_rate
    if final_wave_data is not None:
        return (final_sample_rate, final_wave_data)
    return None

############################################
# Processing Functions
############################################

def process_client_chunk(audio_tuple):
    global client_buffer, min_frames
    if audio_tuple is None:
        return get_final_audio()
    sr, data = audio_tuple
    if data is None:
        return get_final_audio()
    client_buffer.append(data)
    concatenated = np.concatenate(client_buffer)
    if len(concatenated) >= min_frames:
        temp_path = os.path.join(output_dir, f"client_chunk_{time.strftime('%Y%m%d_%H%M%S')}.wav")
        sf.write(temp_path, concatenated, sr)
        process_file(temp_path)
        client_buffer = []
    return get_final_audio()

def process_file(filepath):
    global final_wave_data
    try:
        full_ref_text = transcribe_audio(filepath, language="En")
        print(f"Full Transcribed: {full_ref_text}")
        clipped_filepath = clip_audio(filepath, max_duration_sec=15)
        print(f"Clipped audio saved to: {clipped_filepath}")
        clipped_ref_text = transcribe_audio(clipped_filepath, language="En")
        print(f"Clipped Transcribed: {clipped_ref_text}")
        translated_text = translate_english_to_spanish(full_ref_text)
        print(f"Translated: {translated_text}")
        sr, wave_data = run_tts_inference(
            tts_api,
            ref_audio=clipped_filepath,
            ref_text=clipped_ref_text,
            gen_text=translated_text,
            remove_silence_flag=remove_silence_flag,
            speed_val=speed
        )
        wave_data = wave_data.astype(np.float32)
        update_final_wave(wave_data, sr, crossfade_sec=cross_fade_duration)
        if os.path.exists(clipped_filepath):
            os.remove(clipped_filepath)
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

############################################
# Control Functions for Recording
############################################

def start_recording():
    """Show the microphone input for client recording."""
    return gr.update(visible=True), "Recording started."

def stop_recording():
    """Hide the microphone input to stop recording."""
    return gr.update(visible=False), "Recording stopped."

def reset_translator():
    global final_wave_data, client_buffer
    final_wave_data = None
    client_buffer = []
    return "Translator reset."

############################################
# Gradio Interface
############################################

with gr.Blocks(title="Real-Time Continuous Translation") as app:
    gr.Markdown("# Real-Time Continuous Translation")
    gr.Markdown("This app runs on the GPU‑server and is accessed via a share link. In Orlando, use your microphone to record the sermon and hear the translated Spanish speech on your speakers.")
    
    with gr.Row():
        start_btn = gr.Button("Start Recording")
        stop_btn = gr.Button("Stop Recording")
        reset_btn = gr.Button("Reset")
    
    status_text = gr.Textbox(label="Status", interactive=False)
    
    # Microphone input (hidden by default)
    mic_input = gr.Audio(type="numpy", streaming=True, visible=False)
    
    # Synthesized audio output (auto-plays)
    audio_output = gr.Audio(label="Synthesized Audio", type="numpy", autoplay=True)
    
    # When new audio is received from the mic, process it
    mic_input.change(fn=process_client_chunk, inputs=mic_input, outputs=audio_output)
    
    start_btn.click(fn=start_recording, outputs=[mic_input, status_text])
    stop_btn.click(fn=stop_recording, outputs=[mic_input, status_text])
    reset_btn.click(fn=reset_translator, outputs=status_text)
    
app.launch(share=True)
