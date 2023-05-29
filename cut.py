import wave
import os
import math

def split_wav_file(filename, size_limit=50):
    file_size = os.path.getsize(filename)  # get file size in bytes
    if file_size <= size_limit * 1024 * 1024:  # if file is smaller than limit, do nothing
        return

    wav_file = wave.open(filename, 'rb')
    params = wav_file.getparams()
    bytes_per_sample = params.sampwidth
    samples_per_file = int(size_limit * 1024 * 1024 / bytes_per_sample)

    base_filename, ext = os.path.splitext(filename)
    file_number = 0

    data_frames = wav_file.readframes(samples_per_file)

    while len(data_frames) > 0:  # while there is data to write
        # construct new filename
        new_filename = f"{base_filename}_{file_number}{ext}"
        # write data_frames to new file
        new_wav_file = wave.open(new_filename, 'wb')
        new_wav_file.setparams(params)
        new_wav_file.writeframes(data_frames)
        new_wav_file.close()
        # update file number
        file_number += 1
        # read next set of frames
        data_frames = wav_file.readframes(samples_per_file)

    wav_file.close()



split_wav_file('elmo.wav')
