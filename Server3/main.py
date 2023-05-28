import socket
import collections
import webrtcvad
import struct
import os
import preprocess
import detect_human_voice
import speech_recognition as sr
import text_processing

# Ruta de la carpeta de audio
audio_folder = "audio"

# Obtener la lista de archivos en la carpeta de audio
file_list = os.listdir(audio_folder)

# Eliminar todos los archivos RAW de la carpeta de audio
for file_name in file_list:
    if file_name.endswith(".raw"):
        file_path = os.path.join(audio_folder, file_name)
        os.remove(file_path)

# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific network interface and port number
server_address = ('', 9999)  # '' means all available interfaces
sock.bind(server_address)
sock.listen(10)

sample_rate = 16000
frame_duration_ms = 30  # duration of a frame in ms
frame_length_samples = int(sample_rate * frame_duration_ms / 1000)
frame_length_bytes = frame_length_samples * 2  # 2 bytes per sample

# Create a VAD object
vad = webrtcvad.Vad()
vad.set_mode(3)

# Queue to store frames
frame_queue = collections.deque(maxlen=1)  # store last 10 frames

# Buffer to store incoming audio data
buffer = b''

# File to store the current speech segment
output_file = None
output_file_count = 0

print("Server listening on {}:{}".format(*server_address))

while True:
    # Wait for a connection
    connection, client_address = sock.accept()
    print('Connection from', client_address)

    # Set a timeout for the connection
    connection.settimeout(2)  # 5 seconds

    try:
        # Receive the data
        while True:
            try:
                data = connection.recv(4096)  # 10ms of audio data at 48kHz (1 channel, 2 bytes per sample)
                if not data:
                    print("Client disconnected")
                    break

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
                        output_file = open(f'audio/audio{output_file_count}.raw', 'wb')
                        

                    # If we are currently in a speech segment, write the frame to file
                    if output_file is not None:
                        output_file.write(frame)

                    # Check if we have detected the end of a speech segment
                    if not any(frame_queue) and output_file is not None:
                        print("End of speech detected")
                        output_file.close()
                        output_file = None

                        # Perform preprocessing on the audio (convert format, normalize, etc.)
                        audio_file_path = f'audio/audio{output_file_count}.raw'
                        if detect_human_voice.detect_human_voice(audio_file_path):
                            print("yes")
                            # Perform preprocessing on the audio (convert format, normalize, etc.)
                            #preprocessed_audio = preprocess.preprocess_audio(audio_file_path)
                        else:
                            print("")
                        
                        output_file_count += 1
                        # Recognize speech using the speech recognition module
                        #text = sr.recognize_speech(preprocessed_audio)

                        # Perform further processing or actions based on the recognized text
                        #text_processing.process_text(text)

            except socket.timeout:
                print("Connection timeout")
                break

    except socket.error as e:
        print("Socket error occurred:", e)

    # Close the connection
    connection.close()

sock.close()