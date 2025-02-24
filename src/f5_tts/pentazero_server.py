#!/usr/bin/env python3
import os
import time
import tempfile
import threading
import warnings
import socket
import struct
import json
import base64
import traceback
import numpy as np
import soundfile as sf
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

############################################
# Configuration & Global Variables
############################################

output_dir = "F5-TTS/data/recordings"
processed_dir = "F5-TTS/data/processed"
generated_dir = "F5-TTS/data/generated"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Audio settings (seconds)
min_length = 3       
max_length = 5       
silence_duration = 0.1  
rms_threshold_db = -40.0
min_frames = int(48000 * min_length)
max_frames = int(48000 * max_length)
rms_threshold = 10 ** (rms_threshold_db / 20.0)
max_silence_frames = int(48000 * silence_duration / 512)

# Translation model (Facebook’s NLLB-200 distilled)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True)

# F5-TTS configuration (using your Spanish model)
tts_model_choice = "F5-TTS"
project = "f5-tts_spanish"
ckpt_path = f"F5-TTS/ckpts/{project}/model_last.pt"
remove_silence = False
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

# Global crossfaded audio output
final_wave_data = None
final_sample_rate = 48000

############################################
# Helper Functions
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
                      remove_silence_flag: bool = True, speed_val: float = 1.0) -> tuple:
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

def save_crossfaded_output() -> None:
    if final_wave_data is not None:
        output_wav_path = os.path.join(generated_dir, "crossfaded_full_output.wav")
        sf.write(output_wav_path, final_wave_data, final_sample_rate)
        print(f"Crossfaded output saved to: {output_wav_path}")

############################################
# Slicer and RMS Functions
############################################

def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + (y.strides[axis],)
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

class Slicer:
    def __init__(self, sr: int, threshold: float = -40.0, min_length: int = 2000,
                 min_interval: int = 300, hop_size: int = 20, max_sil_kept: int = 2000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size must hold.")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size must hold.")
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
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
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
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin() + i - self.max_sil_kept
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
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]),
                               0,
                               int(sil_tags[0][0] * self.hop_size)])
            for i in range(len(sil_tags) - 1):
                chunks.append([self._apply_slice(waveform, sil_tags[i][1], sil_tags[i+1][0]),
                               int(sil_tags[i][1] * self.hop_size),
                               int(sil_tags[i+1][0] * self.hop_size)])
            if sil_tags[-1][1] < total_frames:
                chunks.append([self._apply_slice(waveform, sil_tags[-1][1], total_frames),
                               int(sil_tags[-1][1] * self.hop_size),
                               int(total_frames * self.hop_size)])
            return chunks

############################################
# Processing Pipeline Functions
############################################

def process_file(filepath, client_socket):
    """
    Processes a single audio chunk file:
      1. Transcribe full chunk.
      2. Clip audio for TTS reference.
      3. Re-transcribe clipped audio.
      4. Translate transcription.
      5. Run TTS inference.
      6. Send the generated chunk immediately.
      7. Crossfade the chunk into global output.
      8. Save generated files and update directories.
    """
    global final_wave_data
    try:
        full_ref_text = transcribe_audio(filepath, language="En")
        print("Full Transcribed:", full_ref_text)

        clipped_filepath = clip_audio(filepath, max_duration_sec=15)
        print("Clipped audio saved to:", clipped_filepath)

        clipped_ref_text = transcribe_audio(clipped_filepath, language="En")
        print("Clipped Transcribed:", clipped_ref_text)

        translated_text = translate_english_to_spanish(full_ref_text)
        print("Translated:", translated_text)

        sample_rate, wave_data = run_tts_inference(
            tts_api,
            ref_audio=clipped_filepath,
            ref_text=clipped_ref_text,
            gen_text=translated_text,
            remove_silence_flag=remove_silence,
            speed_val=speed
        )
        wave_data = wave_data.astype(np.float32)

        # --- Send this generated chunk immediately ---
        chunk_bytes = struct.pack(f"{len(wave_data)}f", *wave_data)
        header = json.dumps({"type": "chunk", "length": len(wave_data)}) + "\n"
        client_socket.sendall(header.encode("utf-8"))
        client_socket.sendall(chunk_bytes)
        print("Sent a chunk of", len(wave_data), "floats.")

        # --- Update the global crossfaded output ---
        update_final_wave(wave_data, sample_rate, crossfade_sec=cross_fade_duration)

        output_wav_path = os.path.join(generated_dir, f"generated_{os.path.basename(filepath)}")
        sf.write(output_wav_path, wave_data, sample_rate)
        print("Generated audio saved to:", output_wav_path)
        processed_path = os.path.join(processed_dir, os.path.basename(filepath))
        os.rename(filepath, processed_path)
        if os.path.exists(clipped_filepath):
            os.remove(clipped_filepath)
        save_crossfaded_output()
    except Exception as e:
        print("Error processing file", filepath, ":", e)

def process_media(filepath, client_socket):
    """
    Processes an uploaded media file (audio or video):
      - Extracts/converts audio as needed.
      - Uses Slicer to split based on silence.
      - Processes each resulting chunk, sending each to the client.
    """
    try:
        temp_audio_path = None
        if filepath.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            print("Processing video file:", filepath)
            with VideoFileClip(filepath) as video:
                audio = video.audio
                temp_audio_path = os.path.join(output_dir, "temp_extracted_audio.wav")
                audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")
                print("Extracted audio saved to:", temp_audio_path)
        elif filepath.lower().endswith(('.wav', '.mp3', '.flac', '.aac')):
            print("Processing audio file:", filepath)
            if not filepath.lower().endswith('.wav'):
                temp_audio_path = os.path.join(output_dir, "converted_audio.wav")
                from moviepy.audio.io.AudioFileClip import AudioFileClip
                with AudioFileClip(filepath) as audio:
                    audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")
                print("Converted audio saved to:", temp_audio_path)
            else:
                temp_audio_path = filepath
        else:
            print("File format not supported:", filepath)
            return

        try:
            waveform, sr = sf.read(temp_audio_path)
        except Exception as e:
            print(f"sf.read failed: {e}. Using moviepy fallback.")
            fallback_path = os.path.join(output_dir, "converted_audio_fallback.wav")
            with AudioFileClip(temp_audio_path) as audio:
                audio.write_audiofile(fallback_path, fps=48000, nbytes=2, codec="pcm_s16le")
            waveform, sr = sf.read(fallback_path)
            temp_audio_path = fallback_path

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        slicer = Slicer(sr=sr, threshold=-40.0, min_length=3000, min_interval=300, hop_size=20, max_sil_kept=2000)
        sliced_chunks = slicer.slice(waveform)
        if isinstance(sliced_chunks, list):
            for chunk_info in sliced_chunks:
                if isinstance(chunk_info, list) and len(chunk_info) == 3:
                    chunk, start_time, end_time = chunk_info
                    temp_chunk_path = os.path.join(output_dir, f"chunk_{start_time}_{end_time}.wav")
                    sf.write(temp_chunk_path, chunk, sr)
                    process_file(temp_chunk_path, client_socket)
                else:
                    print("Skipping invalid sliced output:", chunk_info)
        else:
            process_file(temp_audio_path, client_socket)
    except Exception as e:
        print("Error processing media file", filepath, ":", e)

############################################
# Socket Server
############################################

def handle_client(client_socket):
    global final_wave_data, final_sample_rate
    try:
        data = b""
        while b"\n" not in data:
            more = client_socket.recv(1024)
            if not more:
                break
            data += more
        if not data:
            return
        payload = json.loads(data.decode("utf-8").strip())
        if "audio" in payload and payload["audio"]:
            audio_b64 = payload["audio"]
            input_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(input_audio_file, "wb") as f:
                f.write(base64.b64decode(audio_b64))
            print("Received input audio file:", input_audio_file)
            final_wave_data = None
            process_media(input_audio_file, client_socket)
            # After all chunks are processed, send the final crossfaded audio.
            if final_wave_data is not None:
                final_bytes = struct.pack(f"{len(final_wave_data)}f", *final_wave_data)
                header = json.dumps({"type": "final", "length": len(final_wave_data)}) + "\n"
                client_socket.sendall(header.encode("utf-8"))
                client_socket.sendall(final_bytes)
                print("Sent final crossfaded audio.")
            client_socket.sendall(b"END_OF_AUDIO")
        else:
            client_socket.sendall(b"Invalid payload: no audio data provided.\n")
    except Exception as e:
        print("Error handling client:", e)
        traceback.print_exc()
    finally:
        client_socket.close()

def start_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    while True:
        client_socket, addr = server.accept()
        print("Accepted connection from", addr)
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9998)
    args = parser.parse_args()
    start_server(args.host, int(args.port))
