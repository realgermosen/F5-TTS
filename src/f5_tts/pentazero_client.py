#!/usr/bin/env python3
import socket
import struct
import json
import base64
import numpy as np
import sounddevice as sd
import threading
import time
import soundfile as sf

# Global playback buffer and lock for thread-safe access
playback_buffer = np.empty((0,), dtype=np.float32)
buffer_lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    global playback_buffer, buffer_lock
    with buffer_lock:
        if len(playback_buffer) >= frames:
            outdata[:, 0] = playback_buffer[:frames]
            playback_buffer = playback_buffer[frames:]
        else:
            n = len(playback_buffer)
            if n > 0:
                outdata[:n, 0] = playback_buffer
            outdata[n:, 0] = 0
            playback_buffer = np.empty((0,), dtype=np.float32)

def read_line(sock):
    """Read bytes from socket until a newline is encountered."""
    line = b""
    while True:
        char = sock.recv(1)
        if not char:
            break
        if char == b"\n":
            break
        line += char
    return line

def run_client(server_ip, server_port, input_audio_path, ref_text="", ref_audio_path_override=None):
    global playback_buffer, buffer_lock

    # Read and encode the input audio file.
    with open(input_audio_path, "rb") as f:
        input_audio_data = f.read()
    input_audio_b64 = base64.b64encode(input_audio_data).decode("utf-8")

    ref_audio_b64 = ""
    if ref_audio_path_override:
        with open(ref_audio_path_override, "rb") as f:
            ref_audio_data = f.read()
        ref_audio_b64 = base64.b64encode(ref_audio_data).decode("utf-8")

    payload = {
        "audio": input_audio_b64,
        "ref_text": ref_text,
        "ref_audio": ref_audio_b64
    }
    payload_str = json.dumps(payload) + "\n"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_ip, int(server_port)))
    s.sendall(payload_str.encode("utf-8"))

    # Start the output stream for immediate playback.
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32', callback=audio_callback)
    stream.start()

    final_crossfade = None  # To hold the downloaded final crossfaded audio

    print("Receiving audio chunks...")
    while True:
        header_line = read_line(s)
        if not header_line:
            break
        if header_line.strip() == b"END_OF_AUDIO":
            break
        try:
            header = json.loads(header_line.decode("utf-8"))
        except Exception as e:
            print("Failed to parse header:", e)
            break
        chunk_type = header.get("type")
        length = header.get("length")
        # Each float is 4 bytes.
        byte_count = length * 4
        chunk_data = b""
        while len(chunk_data) < byte_count:
            more = s.recv(byte_count - len(chunk_data))
            if not more:
                break
            chunk_data += more
        num_floats = len(chunk_data) // 4
        audio_chunk = np.array(struct.unpack(f"{num_floats}f", chunk_data), dtype=np.float32)
        
        if chunk_type == "chunk":
            # Append received chunk to the global playback buffer.
            with buffer_lock:
                playback_buffer = np.concatenate((playback_buffer, audio_chunk))
            print(f"Received {chunk_type} chunk with {num_floats} floats (playing immediately).")
        elif chunk_type == "final":
            # Save final crossfaded audio without adding it to playback.
            final_crossfade = audio_chunk
            print(f"Received final crossfade chunk with {num_floats} floats (downloaded, not played).")
    
    s.close()

    # Wait until the playback buffer has been fully played.
    while True:
        with buffer_lock:
            if len(playback_buffer) == 0:
                break
        time.sleep(0.1)
    stream.stop()
    stream.close()
    print("Playback complete.")

    # Optionally, save the final crossfade to a file if it was received.
    if final_crossfade is not None:
        final_filename = "final_crossfade.wav"
        sf.write(final_filename, final_crossfade, 24000)
        print(f"Final crossfaded audio downloaded and saved to {final_filename}.")

if __name__ == "__main__":
    run_client(
        server_ip="127.0.0.1",  # Update as needed.
        server_port=9998,
        input_audio_path="test audio files/How to get a deeper voice in 30 seconds! #shorts #deepvoice.mp3",
        ref_text="",
        ref_audio_path_override=None
    )
