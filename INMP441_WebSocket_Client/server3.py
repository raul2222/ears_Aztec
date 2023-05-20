import os
import socket
import struct
import webrtcvad
import pydub
import numpy as np

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific network interface and port number
server_address = ('', 9999)  # '' means all available interfaces
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

volume_scale = 1.5

# Create a VAD object
vad = webrtcvad.Vad()

# Set its aggressiveness mode, which is an integer between 0 and 3. 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
vad.set_mode(3)

frame_duration_ms = 30  # frame duration in ms
frame_length_samples = int(48000 * frame_duration_ms / 1000)
frame_length_bytes = frame_length_samples * 2  # 2 bytes per sample

print("Server listening on {}:{}".format(*server_address))

voice_started = False
current_speech = bytearray()
count = 0

ACCUMULATED_FRAMES = 20  # accumulate 20 frames before processing
data_accumulated = []

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)

        # Receive the data
        while True:
            data = connection.recv(frame_length_bytes)
            if data:
                data_accumulated.append(data)
                if len(data_accumulated) < ACCUMULATED_FRAMES:
                    continue  # don't process data until we've accumulated enough frames
                else:
                    data = b''.join(data_accumulated)
                    data_accumulated = []
                    frames = [data[i:i+frame_length_bytes] for i in range(0, len(data), frame_length_bytes)]
                    for frame in frames:
                        pcm_data = bytearray()
                        for i in range(0, len(frame), 2):
                            value = struct.unpack('<h', frame[i:i+2])[0]  # '<h' is for signed 16-bit integer
                            value *= volume_scale
                            value = min(max(value, -32768), 32767)  # Ensure value is within limits
                            pcm_data += struct.pack('<h', int(value))  # Write to file

                        audio = pydub.AudioSegment(
                            data=pcm_data,
                            sample_width=2,
                            frame_rate=48000,  # replace with your audio's sample rate
                            channels=1
                        )

                        try:
                            if vad.is_speech(audio.raw_data, audio.frame_rate):
                                if not voice_started:
                                    voice_started = True
                                current_speech.extend(pcm_data)
                            elif voice_started and len(current_speech) > 0:
                                # End of speech detected
                                with open(f'audio{count}.raw', 'wb') as f:
                                    f.write(bytes(current_speech))
                                count += 1
                                current_speech = bytearray()
                                voice_started = False
                        except:
                            print("Error while processing frame")
                            print("Frame length:", len(audio.raw_data))
                            print("Frame rate:", audio.frame_rate)
                            print("Frame data:", audio.raw_data[:100])  # print the first 100 bytes

            else:
                break
    finally:
        # Clean up the connection
        connection.close()

sock.close()
