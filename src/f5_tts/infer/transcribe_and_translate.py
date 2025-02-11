import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.infer.utils_infer import preprocess_ref_audio_text

# ignore warnings concerning whisper
import warnings

# Suppress specific FutureWarning related to the transformers Whisper model
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="transformers.models.whisper"
)

############################################
# Configuration
############################################

# Directory to monitor for new audio files
input_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
os.makedirs(processed_dir, exist_ok=True)

# Translation model configuration
model_name = "facebook/nllb-200-distilled-600M"
print("Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)

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
    Processes a single audio file: transcribes and translates it.
    """
    try:
        # Transcribe the audio file
        ref_text = preprocess_ref_audio_text(filepath, "", language="En")[1]
        translated_text = translate_text(ref_text)

        # Display transcribed and translated text
        print()
        print(f"Transcribed: {ref_text}")
        print(f"Translated: {translated_text}\n")

        # Move the processed file to a "processed" directory
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
    processed_files = set()  # Keep track of processed files

    while True:
        # Get a list of files in the directory
        files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

        for file in files:
            filepath = os.path.join(input_dir, file)

            # Skip files already processed
            if filepath in processed_files:
                continue

            # Process the file
            process_file(filepath)

            # Mark the file as processed
            processed_files.add(filepath)

        # Wait a bit before checking again
        time.sleep(1)

############################################
# Main Execution
############################################

if __name__ == "__main__":
    watch_directory()
