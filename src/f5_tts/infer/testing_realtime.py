import os
import sounddevice as sd
import soundfile as sf
import queue
import tempfile
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process, load_vocoder

############################################
# Configuration
############################################

output_dir = "F5-TTS/data/realtime_output"
os.makedirs(output_dir, exist_ok=True)

SYSTEM_PROMPT = (
    "You are not an AI assistant, you are an English-to-Spanish translator. "
    "You must produce only the correct and accurate Spanish translation of the given English text, "
    "with no additional commentary, roles, formatting, or extra text. "
    "Keep responses concise since they will be spoken out loud. "
    "Below is the English text that needs to be translated into Spanish:"
)

model_name = "facebook/nllb-200-distilled-600M"
tts_model_choice = "F5-TTS"
project = "penta_common_voice"
tokenizer_type = "char"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
sd.default.device = 0  # Yeti X microphone

############################################
# Initialize Models
############################################

print("Loading vocoder...")
vocoder = load_vocoder()

print("Loading TTS model...")
path_data = os.path.join("F5-TTS/data")
vocab_file = os.path.join(path_data, project + "_" + tokenizer_type, "vocab.txt")

tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)

print("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)

############################################
# Real-time Processing
############################################

def translate_text(english_text = "My name is Carlos"):
    inputs = tokenizer(english_text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id["spa_Latn"]
    translated_tokens = translation_model.generate(
        **inputs, forced_bos_token_id=forced_bos_token_id, max_length=100
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def generate_tts(ref_audio, ref_text, gen_text):
    sample_rate, wave_data = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        tts_api,
        vocoder,
        remove_silence=True,
        cross_fade_duration=0.15,
        speed=1.0,
    )
    return sample_rate, wave_data

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

q = queue.Queue()

def main():
    print("Starting real-time audio processing...")

    # Audio stream setup
    with sd.InputStream(samplerate=48000, channels=1, callback=audio_callback):
        while True:
            # Get audio from the queue
            audio_data = q.get()

            # Write to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                sf.write(temp_wav.name, audio_data, 16000)
                ref_audio_path = temp_wav.name

            # Automatically transcribe the reference audio
            ref_text = preprocess_ref_audio_text(ref_audio_path, "")[1]
            print(f"Transcribed Text: {ref_text}")

            # Translate the transcription
            translated_text = translate_text(ref_text)
            print(f"Translated Text: {translated_text}")

            # Generate TTS audio
            sample_rate, wave_data = generate_tts(ref_audio_path, ref_text, translated_text)

            # Save generated audio
            output_file = os.path.join(output_dir, f"output_{int(time.time())}.wav")
            sf.write(output_file, wave_data, sample_rate)
            print(f"Generated audio saved to: {output_file}")

if __name__ == "__main__":
    main()
