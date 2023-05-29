import os

def rename_files(directory):
    i = 1
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            os.rename(os.path.join(directory, filename), os.path.join(directory, f'{i}.wav'))
            i += 1

# use it like this:
rename_files('/home/raul/ears_Aztec/audio/1/speaker_SPEAKER_02')
