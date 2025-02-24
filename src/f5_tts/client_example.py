import socket
import struct
import numpy as np
import soundfile as sf

def run_client(server_ip, server_port, text, output_wav="output.wav"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((server_ip, int(server_port)))
    
    # Send the text input to the server.
    s.sendall(text.encode("utf-8"))
    
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
    
    # Unpack the received audio data (each float is 4 bytes).
    num_floats = len(audio_data) // 4
    audio_floats = struct.unpack(f"{num_floats}f", audio_data)
    
    # Save the audio as a WAV file. (The server uses a sampling rate of 24000 for streaming.)
    sr = 24000
    sf.write(output_wav, np.array(audio_floats), sr)
    print(f"Saved streamed audio to {output_wav}")

if __name__ == "__main__":
    run_client("127.0.0.1", 9998, "Este es un mensaje de prueba")
