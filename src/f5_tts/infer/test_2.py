import os
import time
import threading
import tempfile
import warnings
import numpy as np
import soundfile as sf
import gradio as gr
import queue

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

# Audio settings (seconds)
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
crossfade_duration = 0.50  
speed = 1.0
vocoder = load_vocoder()
vocab_file = os.path.join("F5-TTS", "data", f"{project}_char", "vocab.txt")
tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)

############################################
# Global Variables for Processing & Playback Queue
############################################

client_buffer = []  # Buffer for incoming client audio
playback_queue = queue.Queue()  # Queue for processed TTS chunks (each: (sample_rate, waveform))

############################################
# Utility Functions
############################################

def transcribe_audio(filepath: str, language: str = "En") -> str:
    _, ref_text = preprocess_ref_audio_text(filepath, "", language=language)
    return ref_text

def translate_english_to_spanish(english_text: str) -> str:
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

def run_tts_inference(tts_api, ref_audio: str, ref_text: str, gen_text: str):
    """Runs text-to-speech inference and returns the generated speech data."""
    seed = -1
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        final_wave, final_sample_rate, _ = tts_api.infer(
            gen_text=gen_text.lower().strip(),
            ref_text=ref_text.lower().strip(),
            ref_file=ref_audio,
            file_wave=f.name,
            speed=speed,
            seed=seed,
            remove_silence=remove_silence_flag,
        )
        sf.write(f.name, final_wave, final_sample_rate)
        return final_sample_rate, final_wave.astype(np.float32)

############################################
# Processing Functions
############################################

def process_file(filepath):
    """Processes a file: transcribe, translate, run TTS, enqueue for playback."""
    try:
        full_ref_text = transcribe_audio(filepath, language="En")
        translated_text = translate_english_to_spanish(full_ref_text)
        sr, wave_data = run_tts_inference(tts_api, ref_audio=filepath, ref_text=full_ref_text, gen_text=translated_text)
        playback_queue.put((sr, wave_data))
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def process_client_chunk(audio_tuple):
    """Processes microphone input and queues it when ready."""
    global client_buffer
    if audio_tuple is None:
        return None
    sr, data = audio_tuple
    client_buffer.append(data)
    concatenated = np.concatenate(client_buffer)
    if len(concatenated) >= min_frames:
        temp_path = os.path.join(output_dir, f"client_chunk_{time.strftime('%Y%m%d_%H%M%S')}.wav")
        sf.write(temp_path, concatenated, sr)
        threading.Thread(target=process_file, args=(temp_path,), daemon=True).start()
        client_buffer = []
    return None

def get_next_chunk():
    """Fetches the next chunk from the queue, or returns None if empty."""
    if not playback_queue.empty():
        return playback_queue.get()
    return None

############################################
# Control Functions for Recording
############################################

def start_recording():
    return gr.update(visible=True), "Recording started."

def stop_recording():
    return gr.update(visible=False), "Recording stopped."

def reset_translator():
    global playback_queue, client_buffer
    with playback_queue.mutex:
        playback_queue.queue.clear()
    client_buffer = []
    return "Translator reset."

############################################
# Gradio Interface
############################################

with gr.Blocks(title="Real-Time Continuous Translation") as app:
    gr.Markdown("# Real-Time Continuous Translation")
    
    with gr.Row():
        start_btn = gr.Button("Start Recording")
        stop_btn = gr.Button("Stop Recording")
        reset_btn = gr.Button("Reset")
    
    status_text = gr.Textbox(label="Status", interactive=False)
    mic_input = gr.Audio(type="numpy", streaming=True, visible=False)
    audio_output = gr.Audio(label="Synthesized Audio", type="numpy", autoplay=True, elem_id="audio_player")
    next_chunk_btn = gr.Button("Next Chunk", visible=False, elem_id="next_chunk_btn")
    
    mic_input.change(fn=process_client_chunk, inputs=mic_input, outputs=[])
    next_chunk_btn.click(fn=get_next_chunk, outputs=audio_output)
    start_btn.click(fn=start_recording, outputs=[mic_input, status_text])
    stop_btn.click(fn=stop_recording, outputs=[mic_input, status_text])
    reset_btn.click(fn=reset_translator, outputs=status_text)
    
    # JavaScript to trigger audio playback sequentially
    auto_refresh_html = gr.HTML(
        """
        <script>
        let observer = new MutationObserver((mutations) => {
            let audioElem = document.getElementById("audio_player");
            if(audioElem && !audioElem.hasAttribute("listener-attached")){
                audioElem.addEventListener("ended", () => document.getElementById("next_chunk_btn").click());
                audioElem.setAttribute("listener-attached", "true");
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
        </script>
        """
    )
    auto_refresh_html

app.launch(share=True)
