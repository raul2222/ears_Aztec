from pydub import AudioSegment, exceptions
import os

def convert_audio_files(input_folder, output_folder):
    # Creamos la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Iteramos sobre todos los archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            try:
                # Cargamos el archivo de audio MP3
                audio = AudioSegment.from_mp3(os.path.join(input_folder, filename))
                
                # Convertimos el archivo a WAV
                audio = audio.set_frame_rate(44100)  # Aseguramos la frecuencia de muestreo a 44.1 kHz
                audio = audio.set_sample_width(2)    # Ajustamos la anchura de la muestra a 2 (16 bits)

                # Guardamos el archivo en la carpeta de salida
                audio.export(os.path.join(output_folder, f"{filename.rsplit('.', 1)[0]}.wav"), format="wav")
            except exceptions.CouldntDecodeError:
                print(f"No se pudo decodificar {filename}. Se omite este archivo.")

# Llamamos a la función
convert_audio_files("audio/sin_pro", "audio/sin_pro")
