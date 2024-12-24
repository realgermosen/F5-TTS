import os
import sys
import time
import soundfile as sf
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    load_vocoder,
)
from importlib.resources import files
import tempfile

############################################
# Configuration
############################################

input_dir = "F5-TTS/data/penta_test_char/wavs"
output_dir = "F5-TTS/data/penta_test_char/generated_wavs"
metadata_path = "F5-TTS/data/penta_test_char/metadata.csv"
os.makedirs(output_dir, exist_ok=True)

SYSTEM_PROMPT = (
    "You are not an AI assistant, you are an English-to-Spanish translator. "
    "You must produce only the correct and accurate Spanish translation of the given English text, "
    "with no additional commentary, roles, formatting, or extra text. "
    "Keep responses concise since they will be spoken out loud. "
    "Below is the English text that needs to be translated into Spanish:"
)

model_name = "Qwen/Qwen2.5-3B-Instruct"
tts_model_choice = "F5-TTS"
project = "penta_common_voice"
tokenizer = "char"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence = True
cross_fade_duration = 0.15  # Original: 0.15
speed = 1.0

############################################
# Utility Functions
############################################

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

print("Loading vocoder...")
vocoder = load_vocoder()

print("Loading F5-TTS model...")
path_data = str(files("f5_tts").joinpath("../../data"))
vocab_file = os.path.join(path_data, project + "_" + tokenizer, "vocab.txt")

tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)

print("Loading chat model...")
# chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
# chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)

@timing_decorator
def generate_response(system_prompt, english_text, model, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": english_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.95,
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    lines = response.split("\n")
    clean_lines = []
    for line in lines:
        lower_line = line.strip().lower()
        if lower_line.startswith("system") or lower_line.startswith("user") or lower_line.startswith("assistant"):
            continue
        clean_lines.append(line.strip())

    final_response = clean_lines[-1] if clean_lines else response
    return final_response

@timing_decorator
def translate(english_text):
        # Input text in English
    article = english_text
    inputs = tokenizer(article, return_tensors="pt")

    # Set the forced_bos_token_id for the target language (Spanish in this case)
    forced_bos_token_id = tokenizer.lang_code_to_id["spa_Latn"] if hasattr(tokenizer, "lang_code_to_id") else tokenizer.convert_tokens_to_ids("spa_Latn")

    # Generate translation
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=forced_bos_token_id, max_length=100
    )
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # Output the translation
    return translation

@timing_decorator
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
# Main Program
############################################

if not os.path.isfile(metadata_path):
    print(f"metadata.csv not found at {metadata_path}")
    sys.exit(1)

with open(metadata_path, "r", encoding="utf-8-sig") as f:
    lines = f.read().strip().split("\n")

for line in lines:
    if not line.strip():
        continue
    parts = line.split("|")
    if len(parts) != 2:
        print(f"Skipping malformed line: {line}")
        continue
    audio_base, english_text = parts

    if not audio_base.endswith(".wav"):
        audio_base += ".wav"

    audio_path = os.path.join(input_dir, audio_base)
    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}, skipping.")
        continue

    print(f"Processing {audio_path}...")
    # translated_text = generate_response(SYSTEM_PROMPT, english_text, chat_model_state, chat_tokenizer_state)
    translated_text = translate(english_text)

    if not translated_text:
        continue

    print("English:", english_text)
    print("Spanish:", translated_text)

    sample_rate, wave_data = run_infer(
        tts_api,
        audio_path,
        english_text,
        translated_text,
        remove_silence,
        cross_fade_duration,
        speed,
    )

    output_wav_path = os.path.join(output_dir, f"generated_{audio_base}")
    sf.write(output_wav_path, wave_data, sample_rate)
    print(f"Saved generated audio to {output_wav_path}")

print("All done!")
