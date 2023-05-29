from audio_utils import normalize_audio, compress_audio
import socket
import collections
import timeit
import time
import os
import torch
import numpy as np
import tempfile
import wave
import threading
import queue
from pydub import AudioSegment
import openai
from langchain.memory import ConversationBufferMemory
from langchain import  LLMChain, PromptTemplate 
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate, play, voices, voice, set_api_key
import signal
from queue import Empty

set_api_key(os.getenv('ELEVEN_API_KEY'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model_id_whisper = 'whisper-1'
model_id = 'gpt-4'

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
                              force_reload=True,onnx=False)

# Obtain necessary functions from utils
get_speech_timestamps, _, read_audio, *_ = utils

sampling_rate = 32000  # also accepts 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('', 9999)
sock.bind(server_address)
sock.listen(10)

FRAME_DURATION_MS = 100
FRAME_LENGTH_SAMPLES = int(sampling_rate * FRAME_DURATION_MS / 1000)
FRAME_LENGTH_BYTES = FRAME_LENGTH_SAMPLES * 2
frame_queue = collections.deque(maxlen=1)
buffer = b''

output_file = None
output_file_count = 0

# Create a queue to buffer incoming data
data_queue = queue.Queue()
post_speech_queue = collections.deque(maxlen=2)

template = """Eres Aztec, un robot que interactua con niños autistas, 
trabajas en una asociacion de niños autistas, proporcionas respuestas cortas,
tienes mucho tacto con lo que dices porque estas rodeado de niños pequeños, 
eres anable, agradable, simpatico, siempre estas dispuesto a ayudar a los demás,


{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0, model_name=model_id, openai_api_key=OPENAI_API_KEY), 
    prompt=prompt, 
    verbose=True, 
    memory=memory,
)

audio_folder = "audio"

accumulated_buffer = b''

# Variable de bloqueo para pausar/descartar el trabajo de write_data_to_file
block_write_data = False

def send_audio_to_whisper(mp3):
    inicio = timeit.default_timer()
    audio_file = open(mp3, "rb")
    response = openai.Audio.transcribe(
        api_key=OPENAI_API_KEY,
        model=model_id_whisper,
        file=audio_file,
        language="es"
    )
    print("Transcripcion Whisper: " + response['text'])
    fin = timeit.default_timer()
    print(f"La función Whisper tardó {fin - inicio} segundos.")
    if response['text'] == "" or response['text'] == "Un poquito más." or response['text'] == "Subtítulos realizados por la comunidad de Amara.org" or response['text'] ==  "¡Gracias por ver el vídeo!":
        return None
    else:
        return response['text']

def play_audio_thread(file_path):
    mp3_file = compress_audio(file_path)
    result_whisper = send_audio_to_whisper(mp3_file)
    if result_whisper != None:
        inicio = timeit.default_timer()
        res_longchain = llm_chain.predict(human_input=result_whisper)
        fin = timeit.default_timer()
        print(f"La función LongChain tardó {fin - inicio} segundos.")

        if res_longchain != None:
            audio = generate(text=res_longchain, voice='6cP8I3tOFFGkwZ7UfTqz', model='eleven_multilingual_v1')
            play(audio)
            print("fin audio *********************")
            time.sleep(2.15)
        # Aquí puedes realizar cualquier acción adicional con el resultado,
        # como guardar la transcripción en una base de datos o procesarla de alguna otra manera

    # Desbloquea el trabajo de write_data_to_file
    global block_write_data
    global accumulated_buffer
    block_write_data = False
    accumulated_buffer = b''

def write_data_to_file():
    global output_file_count  # Make the variable global
    global block_write_data
    wav_file = None
    non_speech_frames = collections.deque(maxlen=1)

    while True:
        accumulated_buffer = data_queue.get()
        # Si block_write_data es True, se descarta el trabajo
        if block_write_data:
            accumulated_buffer = b''
            continue

        # Write the accumulated buffer to a temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
            num_channels = 1  # Set the number of channels
            temp_wav = wave.open(temp_wav_file.name, "wb")
            temp_wav.setnchannels(num_channels)
            temp_wav.setsampwidth(2)  # Assuming 16-bit audio
            temp_wav.setframerate(sampling_rate)
            temp_wav.writeframes(accumulated_buffer)
            temp_wav.close()

            # Read the audio from the temporary wav file
            wav = read_audio(temp_wav_file.name, sampling_rate=sampling_rate)

            # Normalize the audio
            wav = normalize_audio(wav)

            # Get the speech timestamps using Silero VAD model
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
            print(speech_timestamps)
            # Calculate the duration of speech segments
            total_speech_duration = sum(segment['end'] - segment['start'] for segment in speech_timestamps)

            # Calculate the duration of the entire audio clip
            total_duration = len(accumulated_buffer) / sampling_rate

            # Calculate the ratio of speech to total duration
            is_speech = total_speech_duration / total_duration > 0.1
            print(is_speech)
            if is_speech:
                if wav_file is None:
                    file_path = os.path.join(audio_folder, f"output_{output_file_count}.wav")
                    wav_file = wave.open(file_path, "wb")
                    wav_file.setnchannels(num_channels)
                    wav_file.setsampwidth(2)  # Assuming 16-bit audio
                    wav_file.setframerate(sampling_rate)

                wav_file.writeframes(accumulated_buffer)
                non_speech_frames.clear()
            else:
                non_speech_frames.append(accumulated_buffer)
                if wav_file is not None and len(non_speech_frames) == 1:
                    for frame in non_speech_frames:
                        wav_file.writeframes(frame)
                    wav_file.close()
                    ###########################
                    block_write_data = True  # Bloquea el trabajo de write_data_to_file
                    wav_file = None
                    output_file_count += 1
                    non_speech_frames.clear()
                    # Send the file to Whisper for transcription
                    thread = threading.Thread(target=play_audio_thread, args=(file_path,))
                    thread.start()
                    #thread.join()

        # Clear the accumulated buffer
        accumulated_buffer = b''

# Función para manejar la señal de interrupción (Ctrl+C)
def signal_handler(signal, frame):
    global exit_event
    print("Exiting program...")
    exit_event.set()

# Crear el objeto Event para controlar la finalización del programa
exit_event = threading.Event()

# Asociar la función de manejo de señales al evento de interrupción (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# Start the thread for writing data to file
write_data_thread = threading.Thread(target=write_data_to_file)
write_data_thread.start()

# ...
print("Server listening on {}:{}".format(*server_address))

try:
    while not exit_event.is_set():
        connection, client_address = sock.accept()
        print('Connection from', client_address)
        connection.settimeout(5.7)

        while not exit_event.is_set():
            try:
                data = connection.recv(4096)
                if not data:
                    print("Client disconnected")
                    break

                buffer += data

                while len(buffer) >= FRAME_LENGTH_BYTES and not exit_event.is_set():
                    frame = buffer[:FRAME_LENGTH_BYTES]
                    buffer = buffer[FRAME_LENGTH_BYTES:]
                    accumulated_buffer += frame

                    if len(accumulated_buffer) >= sampling_rate * 3:
                        post_speech_queue.append(accumulated_buffer)
                        accumulated_buffer = b''

                        if len(post_speech_queue) == 2:
                            data_queue.put(post_speech_queue.popleft())

            except socket.timeout:
                print("Connection timeout")
                break

        connection.close()
except KeyboardInterrupt:
    print("Server shutting down...")

sock.close()
exit_event.set()
write_data_thread.join()