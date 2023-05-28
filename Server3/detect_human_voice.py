import os
import wave
import webrtcvad
from pydub import AudioSegment
import scipy.fftpack as sf
import numpy as np
from vad import VoiceActivityDetector
import argparse
import json
import whisper
from pydub.playback import play


def detect_human_voice(audio_path):
    """
    Detecta si existe voz humana en el archivo de audio.

    Args:
        audio_path (str): Ruta al archivo de audio a analizar.

    Returns:
        bool: True si se detecta voz humana, False de lo contrario.
    """
    model = whisper.load_model("base")

    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path, sample_width=2, frame_rate=48000, channels=1)

    audio.export(wav_path, format="wav")
    play(audio)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(wav_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    #print(result)
    print(result.no_speech_prob)
    # print the recognized t

    if result.no_speech_prob < 0.75:
        print(result.text)
        return True
    else: 
       
        return False
    




