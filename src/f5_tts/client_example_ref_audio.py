import socket
import struct
import json
import base64
import numpy as np
import soundfile as sf

def run_client(server_ip, server_port, text, ref_text="", ref_audio_path=None, output_wav="output.wav"):
    # If a reference audio file is provided, encode it; otherwise, use an empty string.
    if ref_audio_path:
        with open(ref_audio_path, "rb") as f:
            ref_audio_data = f.read()
        ref_audio_b64 = base64.b64encode(ref_audio_data).decode("utf-8")
    else:
        ref_audio_b64 = ""
    
    # Create JSON payload. ref_text and ref_audio are optional.
    payload = {
        "text": text,
        "ref_text": ref_text if ref_text is not None else "",
        "ref_audio": ref_audio_b64
    }
    payload_str = json.dumps(payload) + "\n"
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_ip, int(server_port)))
    
    # Send JSON payload.
    s.sendall(payload_str.encode("utf-8"))
    
    # Receive audio data until the end-of-audio marker.
    audio_data = b""
    while True:
        chunk = s.recv(4096)
        if b"END_OF_AUDIO" in chunk:
            chunk = chunk.replace(b"END_OF_AUDIO", b"")
            audio_data += chunk
            break
        audio_data += chunk
    s.close()
    
    # Unpack received audio data (each float is 4 bytes).
    num_floats = len(audio_data) // 4
    audio_floats = struct.unpack(f"{num_floats}f", audio_data)
    
    # Save the audio as a WAV file (server streams at 24000 Hz).
    sr = 24000
    sf.write(output_wav, np.array(audio_floats), sr)
    print(f"Saved streamed audio to {output_wav}")

if __name__ == "__main__":
    run_client(
        "127.0.0.1",
        9998,
        "Jesús va a Jerusalén varias veces durante su ministerio, así que tuve que preguntarme: ¿cuál es la razón por la que Jesús va a Jerusalén, en este momento y en esta ocasión?",
        ref_text="Jesus goes to Jerusalem a number of times within his ministry, so I had to ask myself: what is the reason Jesus is going to Jerusalem, in this moment, at this time?",  # Optional reference text.
        ref_audio_path="F5-TTS/data/processed/chunk_8.88_17.3.wav",  # Optional reference audio file; set to a valid path if available.
        output_wav="output.wav"
    )
