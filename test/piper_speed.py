from sunfounder_voice_assistant.tts import Piper
import time

tts = Piper()
tts.set_model('en_US-amy-low')

word = "Hi, I'm piper TTS. A fast and local neural text-to-speech engine that embeds espeak-ng for phonemization."

print("Stream: ", end="", flush=True)
start_time = time.time()
tts.say(word)
end_time = time.time()
print(f"tooks: {end_time - start_time} seconds")


print("None Stream: ", end="", flush=True)
start_time = time.time()
tts.say(word, stream=False)
end_time = time.time()
print(f"tooks: {end_time - start_time} seconds")
