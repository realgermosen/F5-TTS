import os
import time
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav
import tempfile

# Ignore warnings concerning Whisper
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="transformers.models.whisper"
)

############################################
# Configuration
############################################

# Directories
input_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
generated_dir = "F5-TTS/data/generated"  # New directory for generated audios
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Translation model configuration
model_name = "facebook/nllb-200-distilled-600M"
print("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)

# F5-TTS configuration
tts_model_choice = "F5-TTS"
project = "f5-tts_spanish"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence = True
cross_fade_duration = 0.15
speed = 1.0

print("Loading vocoder...")
vocoder = load_vocoder()

print("Loading F5-TTS model...")
path_data = "F5-TTS/data"
vocab_file = os.path.join(path_data, f"{project}_char/vocab.txt")

tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)


def run_infer(tts_api, ref_audio, ref_text, gen_text, remove_silence, cross_fade_duration=0.15, speed=1.0, nfe_step=34):
    # seed = 6839953490746347  # Fixed seed for reproducibility
    # seed = 11111115
    seed = -1
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        final_wave, final_sample_rate, _ = tts_api.infer(
            gen_text=gen_text.lower().strip(),
            ref_text=ref_text.lower().strip(),
            ref_file=ref_audio,
            nfe_step=nfe_step,
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

############################################
# Translation Function
############################################

def translate_text(english_text):
    """
    Translates English text to Spanish using the NLLB model.
    """
    inputs = tokenizer(english_text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id["spa_Latn"] if hasattr(tokenizer, "lang_code_to_id") else tokenizer.convert_tokens_to_ids("spa_Latn")
    translated_tokens = translation_model.generate(
        **inputs, forced_bos_token_id=forced_bos_token_id, max_length=100
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

############################################
# File Processing Logic
############################################

def process_file(filepath):
    """
    Processes a single audio file: transcribes, translates, and generates TTS audio.
    """
    try:
        # Transcribe the audio file
        ref_text = preprocess_ref_audio_text(filepath, "", language="En")[1]
        translated_text = translate_text(ref_text)

        # Display transcription and translation
        print()
        print(f"Transcribed: {ref_text}")
        print(f"Translated: {translated_text}")

        # Generate TTS audio
        print("Generating Spanish TTS audio...")
        sample_rate, wave_data = run_infer(
            tts_api,
            filepath,
            ref_text,
            translated_text,
            remove_silence,
            cross_fade_duration,
            speed,
        )

        # Save generated audio to the new directory
        output_wav_path = os.path.join(generated_dir, f"generated_{os.path.basename(filepath)}")
        sf.write(output_wav_path, wave_data, sample_rate)
        print(f"Generated audio saved to: {output_wav_path}")

        # Move original processed file
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)

        return ref_text, translated_text

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None

############################################
# Directory Watcher
############################################

def watch_directory():
    """
    Monitors the input directory for new audio files and processes them in order.
    """
    print("Watching directory for new audio files...")
    processed_files = set()

    while True:
        # Get all WAV files in the directory
        files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

        for file in files:
            filepath = os.path.join(input_dir, file)

            # Skip already processed files
            if filepath in processed_files:
                continue

            # Process the file
            process_file(filepath)

            # Mark file as processed
            processed_files.add(filepath)

        time.sleep(1)

############################################
# Main Execution
############################################

if __name__ == "__main__":
    watch_directory()
