from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import math

PIANNOTE_KEY = os.getenv('PIANNOTE_KEY')

# 1. instantiate pretrained speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=PIANNOTE_KEY)

# 2. apply pretrained pipeline
diarization = pipeline("audio/sin_pro/onlymp3.wav")

# 3. load the entire audio file
audio = AudioSegment.from_mp3("audio/sin_pro/onlymp3.wav")

# 4. prepare a dictionary to hold audio for each speaker
speakers = {}

# 5. iterate over each detected speaker segment
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # convert times to milliseconds
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)
    
    # only process segments longer than 9 seconds
    if end_ms - start_ms > 9000:
        # calculate the number of segments that this duration can be divided into
        num_segments = math.ceil((end_ms - start_ms) / 15000)
        
        for i in range(num_segments):
            segment_start_ms = start_ms + i * 15000
            segment_end_ms = min(start_ms + (i + 1) * 15000, end_ms)
            
            if segment_end_ms - segment_start_ms > 9000:
                segment = audio[segment_start_ms:segment_end_ms]
                
                if speaker in speakers:
                    speakers[speaker].append(segment)
                else:
                    speakers[speaker] = [segment]

# 6. save each speaker's audio to a separate file
for speaker, segments in speakers.items():
    # create a directory for the speaker if it doesn't exist
    speaker_dir = f"audio/speaker_{speaker}"
    os.makedirs(speaker_dir, exist_ok=True)
    
    # save each speaker's audio to their respective directory
    for i, segment in enumerate(segments):
        segment.export(os.path.join(speaker_dir, f"speaker_{speaker}_{i}.wav"), format="wav")