from faster_whisper import WhisperModel
import time
model_size = "medium"


model = WhisperModel(model_size, device="cpu", compute_type="int8")



print("ya")
segments, info = model.transcribe("/home/raul/ears_Aztec/audio/1/speaker_SPEAKER_02/1.wav", beam_size=5)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print("ya")
segments, info = model.transcribe("output_0.mp3", beam_size=5)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


"""
segments = info = ""
time.sleep(3)
print("ya")
segments, info = model.transcribe("output_0.mp3", beam_size=5)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

"""