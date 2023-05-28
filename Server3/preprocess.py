from pydub import AudioSegment
import os
from pydub.playback import play
import wave
import sys

from vosk import Model, KaldiRecognizer, SetLogLevel


def preprocess_audio(audio_path):
    """
    Realiza el preprocesamiento del audio antes de la transcripción.

    Args:
        audio_path (str): Ruta al archivo de audio a preprocesar.

    Returns:
        bytes: Audio preprocesado en el formato adecuado para el reconocimiento de voz a texto.
    """


    # You can set log level to -1 to disable debug messages
    SetLogLevel(-1)

    # Convertir el archivo de audio a formato WAV utilizando pydub
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path, sample_width=2, frame_rate=48000, channels=1)


    audio.export(wav_path, format="wav")

    # Reproducir el audio utilizando pydub
    play(audio)

    # Cargar el modelo de Vosk
    model = Model("/home/raul/ears_Aztec/Server3/model/vosk-model-small-es-0.42")

    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("El archivo de audio debe estar en formato WAV mono PCM.")
        sys.exit(1)

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print(rec.Result())
        else:
            print(rec.PartialResult())

    print(rec.FinalResult())

    # Realizar el preprocesamiento adicional si es necesario
    # ...

    # Leer el archivo de audio y devolver los datos preprocesados
    with open(audio_path, "rb") as file:
        audio_bytes = file.read()

    return audio_bytes