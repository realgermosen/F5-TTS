#!/usr/bin/env python3
import asyncio
import websockets
import json
import struct
import numpy as np
import soundfile as sf
import tempfile
import os
import time
import warnings
import scipy.signal  # for resampling if needed

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, load_vocoder, remove_silence_for_generated_wav
from moviepy.audio.io.AudioFileClip import AudioFileClip

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")

############################################
# Configuration & Global Variables
############################################
# All processing is now at 24000 Hz.
SAMPLE_RATE = 24000

output_dir = "data/recordings"
processed_dir = "data/processed"
generated_dir = "data/generated"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

silence_threshold_db = -40.0
SILENCE_THRESHOLD = 10 ** (silence_threshold_db / 20.0)

# Timing parameters (in seconds)
MIN_LENGTH_SEC = 4.0       # Minimum segment duration before processing
MAX_LENGTH_SEC = 8.0       # Maximum segment duration; if exceeded, set waiting flag
MAX_REF_DURATION_SEC = 10.0 # Clip reference audio to this duration (ideal for TTS)
SILENCE_REQUIRED_SEC = 0.5  # Duration of silence required before slicing

# Translation model
model_name = "facebook/nllb-200-distilled-600M"
print("[SERVER DEBUG] Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, src_lang="eng_Latn")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True)
print("[SERVER DEBUG] Translation model loaded.")

# F5-TTS configuration
tts_model_choice = "F5-TTS"
project = "f5-tts_spanish"
ckpt_path = f"ckpts/{project}/model_1200000.safetensors"
remove_silence = False
cross_fade_duration = 0.15  # seconds
speed = 1.0
print("[SERVER DEBUG] Loading vocoder and TTS model...")
vocoder = load_vocoder()
vocab_file = os.path.join("data", f"{project}_char/vocab.txt")
tts_api = F5TTS(
    model_type=tts_model_choice,
    ckpt_file=ckpt_path,
    vocab_file=vocab_file,
    device="cuda",
    use_ema=True,
)
print("[SERVER DEBUG] F5-TTS model loaded.")

final_wave_data = None  # Global accumulated final crossfade

############################################
# Slicer & RMS Functions
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
        # All length parameters are in milliseconds.
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
        if waveform.ndim > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        print("[SERVER DEBUG] Slicing waveform; shape:", waveform.shape)
        if waveform.ndim > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            print("[SERVER DEBUG] Waveform too short; returning original.")
            return [waveform]
        rms_list = get_rms(samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
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
            need_slice = (i - silence_start >= self.min_interval) and (i - clip_start >= self.min_length)
            if need_slice:
                pos = rms_list[silence_start : i+1].argmin() + silence_start
                sil_tags.append((pos, pos))
                clip_start = pos
            silence_start = None
        total_frames = rms_list.shape[0]
        if len(sil_tags) == 0:
            print("[SERVER DEBUG] No silence found; returning entire waveform.")
            return [[waveform, 0, int(total_frames * self.hop_size)]]
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
        print(f"[SERVER DEBUG] Slicer produced {len(chunks)} chunks.")
        return chunks

############################################
# Pipeline Functions (Transcription, Translation, TTS)
############################################
def transcribe_audio(filepath: str, language: str = "En") -> str:
    print(f"[SERVER DEBUG] Transcribing audio from file: {filepath}")
    _, ref_text = preprocess_ref_audio_text(filepath, "", language=language)
    print(f"[SERVER DEBUG] Transcription result: {ref_text}")
    return ref_text

def translate_english_to_spanish(english_text: str) -> str:
    print(f"[SERVER DEBUG] Translating text: {english_text}")
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
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"[SERVER DEBUG] Translation result: {result}")
    return result

def clip_audio(input_filepath: str, max_duration_sec: int = MAX_REF_DURATION_SEC, output_filepath: str = None) -> str:
    print(f"[SERVER DEBUG] Clipping audio from {input_filepath} to {max_duration_sec} sec")
    data, sr = sf.read(input_filepath)
    if data.ndim > 1:
        data = data[:, 0]
    max_samples = int(max_duration_sec * sr)
    if len(data) > max_samples:
        clipped_data = data[:max_samples]
    else:
        clipped_data = data
    if output_filepath is None:
        base = os.path.basename(input_filepath)
        output_filepath = os.path.join(output_dir, f"clipped_{base}")
    sf.write(output_filepath, clipped_data, sr)
    print(f"[SERVER DEBUG] Clipped audio saved to {output_filepath}")
    return output_filepath

def run_tts_inference(tts_api, ref_audio: str, ref_text: str, gen_text: str,
                      remove_silence_flag: bool = True, speed_val: float = 1.0) -> tuple:
    print("[SERVER DEBUG] Running TTS inference...")
    print(f"[SERVER DEBUG] Reference audio: {ref_audio}")
    print(f"[SERVER DEBUG] Reference text: {ref_text}")
    print(f"[SERVER DEBUG] Generation text: {gen_text}")
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
    print(f"[SERVER DEBUG] TTS inference complete. Generated {len(final_wave)} samples at {final_sample_rate} Hz.")
    return final_sample_rate, final_wave

def update_final_wave(new_chunk: np.ndarray, sr: int, crossfade_sec: float = cross_fade_duration) -> None:
    global final_wave_data
    print(f"[SERVER DEBUG] Updating final wave with new chunk of {len(new_chunk)} samples.")
    if final_wave_data is None:
        final_wave_data = new_chunk
    else:
        final_wave_data = np.concatenate((final_wave_data, new_chunk))
    print(f"[SERVER DEBUG] Final wave updated; current length: {len(final_wave_data)} samples.")

############################################
# Process Audio Buffer Function
############################################
def process_audio_buffer(audio_buffer: np.ndarray) -> tuple:
    print(f"[SERVER DEBUG] Processing audio buffer of {len(audio_buffer)} samples ({len(audio_buffer)/SAMPLE_RATE:.2f} sec).")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(temp_file, audio_buffer, SAMPLE_RATE)
    print(f"[SERVER DEBUG] Temporary file written: {temp_file}")
    try:
        full_ref_text = transcribe_audio(temp_file, language="En")
        if not full_ref_text.strip():
            print("[SERVER DEBUG] No transcription available (silence). Skipping segment.")
            return None, None, None
        clipped_filepath = clip_audio(temp_file, max_duration_sec=MAX_REF_DURATION_SEC)
        clipped_ref_text = transcribe_audio(clipped_filepath, language="En")
        if not clipped_ref_text.strip():
            print("[SERVER DEBUG] No clipped transcription available. Skipping segment.")
            return None, None, None
        translated_text = translate_english_to_spanish(full_ref_text)
        sr_out, tts_chunk = run_tts_inference(tts_api, clipped_filepath, clipped_ref_text, translated_text,
                                               remove_silence_flag=remove_silence, speed_val=speed)
        update_final_wave(tts_chunk, sr_out, crossfade_sec=cross_fade_duration)
        if os.path.exists(clipped_filepath):
            os.remove(clipped_filepath)
        info = {
            "transcription": full_ref_text,
            "clipped": clipped_ref_text,
            "translation": translated_text
        }
        print("[SERVER DEBUG] Audio buffer processed successfully.")
        return tts_chunk, sr_out, info
    except Exception as e:
        print("[SERVER DEBUG] Error processing audio buffer:", e)
        return None, None, None
    finally:
        os.remove(temp_file)
        print(f"[SERVER DEBUG] Temporary file {temp_file} removed.")

############################################
# Silence Detection Function
############################################
def is_silence(audio_chunk: np.ndarray, duration_sec: float = SILENCE_REQUIRED_SEC, threshold: float = SILENCE_THRESHOLD) -> bool:
    required_samples = int(duration_sec * SAMPLE_RATE)
    if len(audio_chunk) < required_samples:
        return False
    segment = audio_chunk[-required_samples:]
    rms = np.sqrt(np.mean(segment**2))
    print(f"[SERVER DEBUG] Silence detection: RMS over last {duration_sec} sec = {rms:.5f}")
    return rms < threshold

############################################
# File Upload Handling
############################################
file_buffers = {}  # websocket -> bytes

async def process_uploaded_file(websocket, file_bytes: bytes):
    print(f"[SERVER DEBUG] Processing uploaded file of size {len(file_bytes)} bytes.")
    temp_filename = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
    print(f"[SERVER DEBUG] Uploaded file written to temporary file: {temp_filename}")
    try:
        data, sr = sf.read(temp_filename)
        print(f"[SERVER DEBUG] Read uploaded file: {len(data)} samples at {sr} Hz.")
    except Exception as e:
        print(f"[SERVER DEBUG] sf.read failed: {e}. Using fallback conversion.")
        fallback_filename = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with AudioFileClip(temp_filename) as audio:
            audio.write_audiofile(fallback_filename, fps=SAMPLE_RATE, nbytes=2, codec="pcm_s16le")
        data, sr = sf.read(fallback_filename)
        os.remove(fallback_filename)
        print(f"[SERVER DEBUG] Fallback conversion succeeded: {len(data)} samples at {sr} Hz.")
    os.remove(temp_filename)
    print(f"[SERVER DEBUG] Temporary uploaded file removed.")
    if sr != SAMPLE_RATE:
        print(f"[SERVER DEBUG] Resampling uploaded file from {sr} Hz to {SAMPLE_RATE} Hz.")
        new_length = int(len(data) * SAMPLE_RATE / sr)
        data = scipy.signal.resample(data, new_length)
        sr = SAMPLE_RATE
        print(f"[SERVER DEBUG] Resampling complete: new length {len(data)} samples.")
    if data.ndim > 1:
        data = data.mean(axis=1)
    slicer = Slicer(sr=sr, threshold=silence_threshold_db, min_length=int(MIN_LENGTH_SEC*1000),
                    min_interval=300, hop_size=20, max_sil_kept=2000)
    segments = slicer.slice(data)
    print(f"[SERVER DEBUG] Uploaded file segmented into {len(segments)} chunks.")
    for seg in segments:
        if isinstance(seg, list) and len(seg) == 3:
            segment_data = seg[0]
        else:
            segment_data = seg
        tts_chunk, sr_out, info = process_audio_buffer(segment_data)
        if tts_chunk is not None:
            header = {
                "type": "chunk",
                "length": len(tts_chunk),
                "transcription": info.get("transcription", ""),
                "clipped": info.get("clipped", ""),
                "translation": info.get("translation", "")
            }
            header_str = json.dumps(header) + "\n"
            await websocket.send(header_str)
            await websocket.send(tts_chunk.tobytes())
            print(f"[SERVER DEBUG] Sent TTS chunk from uploaded file: {len(tts_chunk)} samples.")
            await asyncio.sleep(0.5)
    if final_wave_data is not None:
        header = json.dumps({"type": "final", "length": len(final_wave_data)}) + "\n"
        await websocket.send(header)
        await websocket.send(final_wave_data.tobytes())
        print(f"[SERVER DEBUG] Sent final crossfade: {len(final_wave_data)} samples.")
    await websocket.send("END_OF_AUDIO")
    print("[SERVER DEBUG] Finished processing uploaded file.")

############################################
# WebSocket Server Handler
############################################
# For each client we store: (audio buffer, last update time, mode, waiting_flag)
# Mode "live": raw float32 streaming; mode "upload": accumulating file bytes.
client_buffers = {}  # websocket -> (np.array, last time, mode, waiting_flag)
file_buffers = {}    # websocket -> bytes

async def process_client(websocket, path=9998):
    global client_buffers, final_wave_data, file_buffers
    print("[SERVER DEBUG] Client connected")
    client_buffers[websocket] = (np.empty((0,), dtype=np.float32), time.time(), "live", False)
    file_buffers[websocket] = b""
    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    print(f"[SERVER DEBUG] Received command: {data}")
                except Exception as e:
                    print("[SERVER DEBUG] Failed to parse text message:", e)
                    continue
                command = data.get("command")
                if command == "stop":
                    buffer, _, mode, _ = client_buffers[websocket]
                    print(f"[SERVER DEBUG] Received stop command in mode: {mode}")
                    if mode == "live" and len(buffer) >= MIN_LENGTH_SEC * SAMPLE_RATE:
                        tts_chunk, sr_out, info = process_audio_buffer(buffer)
                        if tts_chunk is not None:
                            header = {
                                "type": "chunk",
                                "length": len(tts_chunk),
                                "transcription": info.get("transcription", ""),
                                "clipped": info.get("clipped", ""),
                                "translation": info.get("translation", "")
                            }
                            await websocket.send(json.dumps(header) + "\n")
                            await websocket.send(tts_chunk.tobytes())
                    if final_wave_data is not None:
                        header = json.dumps({"type": "final", "length": len(final_wave_data)}) + "\n"
                        await websocket.send(header)
                        await websocket.send(final_wave_data.tobytes())
                    await websocket.send("END_OF_AUDIO")
                    print("[SERVER DEBUG] Stop command processed.")
                    break
                elif command == "upload":
                    client_buffers[websocket] = (np.empty((0,), dtype=np.float32), time.time(), "upload", False)
                    file_buffers[websocket] = b""
                    print("[SERVER DEBUG] Switched to file upload mode.")
                elif command == "upload_end":
                    print("[SERVER DEBUG] File upload complete. Processing file...")
                    await process_uploaded_file(websocket, file_buffers[websocket])
                    client_buffers[websocket] = (np.empty((0,), dtype=np.float32), time.time(), "live", False)
                    file_buffers[websocket] = b""
                    break
            elif isinstance(message, bytes):
                _, _, mode, waiting = client_buffers[websocket]
                if mode == "upload":
                    file_buffers[websocket] += message
                    print(f"[SERVER DEBUG] Received {len(message)} bytes for file upload (total: {len(file_buffers[websocket])} bytes).")
                else:
                    new_data = np.frombuffer(message, dtype=np.float32)
                    buffer, last_time, mode, waiting = client_buffers[websocket]
                    buffer = np.concatenate((buffer, new_data))
                    client_buffers[websocket] = (buffer, time.time(), mode, waiting)
                    duration = len(buffer) / SAMPLE_RATE
                    if duration < MIN_LENGTH_SEC:
                        continue
                    if not waiting and duration >= MAX_LENGTH_SEC:
                        waiting = True
                        client_buffers[websocket] = (buffer, time.time(), mode, waiting)
                        print("[SERVER DEBUG] Buffer exceeded MAX_LENGTH_SEC; waiting for silence.")
                    if waiting and is_silence(buffer, duration_sec=SILENCE_REQUIRED_SEC):
                        print("[SERVER DEBUG] Waiting flag set and silence detected; processing live segment.")
                        tts_chunk, sr_out, info = process_audio_buffer(buffer)
                        if tts_chunk is not None:
                            header = {
                                "type": "chunk",
                                "length": len(tts_chunk),
                                "transcription": info.get("transcription", ""),
                                "clipped": info.get("clipped", ""),
                                "translation": info.get("translation", "")
                            }
                            await websocket.send(json.dumps(header) + "\n")
                            await websocket.send(tts_chunk.tobytes())
                        client_buffers[websocket] = (np.empty((0,), dtype=np.float32), time.time(), mode, False)
                    elif not waiting and is_silence(buffer, duration_sec=SILENCE_REQUIRED_SEC):
                        print("[SERVER DEBUG] Silence detected; processing live segment immediately.")
                        tts_chunk, sr_out, info = process_audio_buffer(buffer)
                        if tts_chunk is not None:
                            header = {
                                "type": "chunk",
                                "length": len(tts_chunk),
                                "transcription": info.get("transcription", ""),
                                "clipped": info.get("clipped", ""),
                                "translation": info.get("translation", "")
                            }
                            await websocket.send(json.dumps(header) + "\n")
                            await websocket.send(tts_chunk.tobytes())
                        client_buffers[websocket] = (np.empty((0,), dtype=np.float32), time.time(), mode, False)
    except Exception as e:
        print("[SERVER DEBUG] Error in client connection:", e)
    finally:
        client_buffers.pop(websocket, None)
        file_buffers.pop(websocket, None)
        print("[SERVER DEBUG] Client disconnected")

############################################
# Start WebSocket Server
############################################
async def main():
    async with websockets.serve(process_client, "0.0.0.0", 9998):
        print("[SERVER DEBUG] WebSocket server listening on 0.0.0.0:9998")
        await asyncio.Future()  # run forever

def run_server():
    asyncio.run(main())  # Ensure the coroutine runs properly

if __name__ == "__main__":
    run_server()
