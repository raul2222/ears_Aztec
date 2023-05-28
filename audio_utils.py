# audio_utils.py
import pygame
import torch
from pydub import AudioSegment
import pygame
import os

API_KEY = "your_api_key_here"
model_id = 'whisper-1'

def play_wav_file(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def normalize_audio(audio):
    audio = audio.float()
    audio /= torch.max(torch.abs(audio))
    return audio


def compress_audio(file_path):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    # Compress the audio to MP3 format with a bitrate of 128 kbps
    mp3_file = os.path.splitext(file_path)[0] + ".mp3"
    audio.export(mp3_file, format="mp3", bitrate="128k")
    return mp3_file


audio_folder = "audio"
file_list = os.listdir(audio_folder)
accumulated_buffer = b''
for file_name in file_list:
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        os.remove(file_path)
