import argparse
import gc
import socket
import struct
import torch
import torchaudio
import traceback
import json
import base64
from threading import Thread
import tempfile

# Try to import 'files' from importlib.resources; if not available, try importlib_resources.
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from cached_path import cached_path

from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from model.backbones.dit import DiT


class TTSStreamingProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load the model using the provided checkpoint and vocab files.
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",  # or "bigvgan"
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

        # Load the vocoder.
        self.vocoder = load_vocoder(is_local=False)

        # Set sampling rate for streaming.
        self.sampling_rate = 24000  # Consistency with client

        # Default reference audio and text.
        self.ref_audio = ref_audio
        self.ref_text = ref_text

        self._warm_up()

    def _warm_up(self):
        print("Warming up the model...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
        audio, sr = torchaudio.load(ref_audio)
        gen_text = "Warm-up text for the model."
        infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
        print("Warm-up completed.")

    def generate_stream(self, text, play_steps_in_s=0.5, ref_audio_override=None, ref_text_override=None):
        """
        Generate audio in chunks and yield them.
        If ref_audio_override or ref_text_override are provided (and non-empty),
        they are used instead of the defaults.
        """
        print("DEBUG: Received synthesis text:", text)
        ref_audio_file = ref_audio_override if ref_audio_override not in [None, ""] else self.ref_audio
        ref_text_val = ref_text_override if ref_text_override not in [None, ""] else self.ref_text
        ref_audio_file, ref_text_val = preprocess_ref_audio_text(ref_audio_file, ref_text_val)
        print("DEBUG: Using reference text:", ref_text_val)
        audio, sr = torchaudio.load(ref_audio_file)
        audio_chunk, final_sample_rate, _ = infer_batch_process(
            (audio, sr),
            ref_text_val,
            [text],
            self.model,
            self.vocoder,
            device=self.device,
        )
        print("DEBUG: Generated audio chunk length:", len(audio_chunk))
        chunk_size = int(final_sample_rate * play_steps_in_s)
        if len(audio_chunk) < chunk_size:
            packed_audio = struct.pack(f"{len(audio_chunk)}f", *audio_chunk)
            yield packed_audio
            return
        for i in range(0, len(audio_chunk), chunk_size):
            chunk = audio_chunk[i : i + chunk_size]
            if i + chunk_size >= len(audio_chunk):
                chunk = audio_chunk[i:]
            if len(chunk) > 0:
                packed_audio = struct.pack(f"{len(chunk)}f", *chunk)
                yield packed_audio


def handle_client(client_socket, processor):
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
        text = payload.get("text") or ""
        ref_text = payload.get("ref_text") or ""
        ref_audio_b64 = payload.get("ref_audio") or ""
        print("DEBUG: Received payload:", payload)
        ref_audio_file = None
        if ref_audio_b64:
            ref_audio_data = base64.b64decode(ref_audio_b64)
            ref_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(ref_audio_file, "wb") as f:
                f.write(ref_audio_data)
            print("DEBUG: Saved override ref_audio to:", ref_audio_file)
        else:
            print("DEBUG: No override ref_audio provided; using default.")
        for audio_chunk in processor.generate_stream(text, ref_audio_override=ref_audio_file, ref_text_override=ref_text):
            client_socket.sendall(audio_chunk)
        client_socket.sendall(b"END_OF_AUDIO")
    except Exception as e:
        print("DEBUG: Error handling client:", e)
        traceback.print_exc()
    finally:
        client_socket.close()


def start_server(host, port, processor):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(port)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = Thread(target=handle_client, args=(client_socket, processor))
        client_handler.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9998)
    parser.add_argument(
        "--ckpt_file",
        default=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")),
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vocab_file",
        default="",
        help="Path to the vocab file if customized",
    )
    parser.add_argument(
        "--ref_audio",
        default=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        help="Default reference audio (used if no override provided)",
    )
    parser.add_argument(
        "--ref_text",
        default="",
        help="Default reference text (used if no override provided)",
    )
    parser.add_argument("--device", default=None, help="Device to run the model on")
    parser.add_argument("--dtype", default=torch.float32, help="Data type to use for model inference")
    args = parser.parse_args()
    try:
        processor = TTSStreamingProcessor(
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            device=args.device,
            dtype=args.dtype,
        )
        start_server(args.host, args.port, processor)
    except KeyboardInterrupt:
        gc.collect()
