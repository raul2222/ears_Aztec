import socket
import collections
import webrtcvad
import struct
import os

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific network interface and port number
server_address = ('', 9999)  # '' means all available interfaces
sock.bind(server_address)
sock.listen(1)

sample_rate = 48000
frame_duration_ms = 30  # duration of a frame in ms
frame_length_samples = int(sample_rate * frame_duration_ms / 1000)
frame_length_bytes = frame_length_samples * 2  # 2 bytes per sample

# Create a VAD object
vad = webrtcvad.Vad()
vad.set_mode(3)

# Queue to store frames
frame_queue = collections.deque(maxlen=10)  # store last 10 frames

# Buffer to store incoming audio data
buffer = b''

# File to store the current speech segment
output_file = None
output_file_count = 0

print("Server listening on {}:{}".format(*server_address))

while True:
    # Wait for a connection
    connection, client_address = sock.accept()
    print('connection from', client_address)

    # Receive the data
    while True:
        data = connection.recv(960)  # 10ms of audio data at 48kHz (1 channel, 2 bytes per sample)
        print("Frame data:", data[:50])
        if data:
            buffer += data
            while len(buffer) >= frame_length_bytes:
                frame = buffer[:frame_length_bytes]
                buffer = buffer[frame_length_bytes:]

                # Detect voice in the frame
                is_speech = vad.is_speech(frame, sample_rate)

                # Add the result to the queue
                frame_queue.append(is_speech)

                # Check if we have detected the start of a speech segment
                if all(frame_queue) and output_file is None:
                    print("Start of speech detected")
                    output_file = open(f'audio{output_file_count}.raw', 'wb')
                    output_file_count += 1

                # If we are currently in a speech segment, write the frame to file
                if output_file is not None:
                    output_file.write(frame)

                # Check if we have detected the end of a speech segment
                if not any(frame_queue) and output_file is not None:
                    print("End of speech detected")
                    output_file.close()
                    output_file = None

        else:
            break

    connection.close()

sock.close()
