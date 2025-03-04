#!/usr/bin/env python3
import socket
import struct
import json
import base64
import numpy as np
import soundfile as sf

def run_client(server_ip, server_port, input_audio_path, ref_text="", ref_audio_path_override=None, output_wav="output.wav"):
    # Read and base64-encode the input audio file.
    with open(input_audio_path, "rb") as f:
        input_audio_data = f.read()
    input_audio_b64 = base64.b64encode(input_audio_data).decode("utf-8")
    
    # Optionally encode a reference audio override.
    ref_audio_b64 = ""
    if ref_audio_path_override:
        with open(ref_audio_path_override, "rb") as f:
            ref_audio_data = f.read()
        ref_audio_b64 = base64.b64encode(ref_audio_data).decode("utf-8")
    
    # Build JSON payload.
    payload = {
        "audio": input_audio_b64,
        "ref_text": ref_text,
        "ref_audio": ref_audio_b64
    }
    payload_str = json.dumps(payload) + "\n"
    
    # Connect to the server and send the payload.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_ip, int(server_port)))
    s.sendall(payload_str.encode("utf-8"))
    
    # Receive streamed audio data until the "END_OF_AUDIO" marker.
    audio_data = b""
    while True:
        chunk = s.recv(4096)
        if b"END_OF_AUDIO" in chunk:
            chunk = chunk.replace(b"END_OF_AUDIO", b"")
            audio_data += chunk
            break
        audio_data += chunk
    s.close()
    
    # Unpack the float32 audio data.
    num_floats = len(audio_data) // 4
    audio_floats = struct.unpack(f"{num_floats}f", audio_data)
    
    # Save the received audio to a WAV file (sample rate 48000).
    sr = 24000
    sf.write(output_wav, np.array(audio_floats), sr)
    print(f"Saved streamed audio to {output_wav}")

if __name__ == "__main__":
    # Example usage: update the paths as needed.
    run_client(
        server_ip="127.0.0.1",
        server_port=9998,
        input_audio_path="test audio files/How to get a deeper voice in 30 seconds! #shorts #deepvoice.mp3",  # The audio file to process.
        ref_text="",   # Optional reference text.
        ref_audio_path_override=None,  # Optional override reference audio.
        output_wav="output.wav"
    )
