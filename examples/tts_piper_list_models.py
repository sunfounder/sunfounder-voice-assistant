"""List all available Piper TTS models.

Run this when you want to check for new models online:
    python examples/tts_piper_list_models.py
"""
from sunfounder_voice_assistant.tts import Piper

tts = Piper()
print(f"Before update: {len(tts.available_models())} models")
print()

tts.update_model_list()

print(f"After update: {len(tts.available_models())} models")
print()
for country in tts.available_countrys():
    models = tts.available_models(country)
    print(f"  {country}: {models}")
