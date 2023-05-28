from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
PIANNOTE_KEY = os.getenv('PIANNOTE_KEY')
# 1. instantiate pretrained speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=PIANNOTE_KEY)

# 2. apply pretrained pipeline
diarization = pipeline("audio/jar.mp3")

# 3. load the entire audio file
audio = AudioSegment.from_mp3("audio/jar.mp3")

# 4. prepare a dictionary to hold audio for each speaker
speakers = {}

# 5. iterate over each detected speaker segment
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # convert times to milliseconds
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)

    # extract this bit of audio
    segment = audio[start_ms:end_ms]

    # if we've seen this speaker before, append the segment
    # otherwise, start a new segment
    if speaker in speakers:
        speakers[speaker] += segment
    else:
        speakers[speaker] = segment

# 6. save each speaker's audio to a separate file
for speaker, audio in speakers.items():
    audio.export(f"speaker_{speaker}.mp3", format="mp3")
