"""List all available Vosk STT models.

Run this when you want to check for new models online:
    python examples/stt_vosk_list_models.py
"""
from sunfounder_voice_assistant.stt import STT

stt = STT()
print(f"Before update: {len(stt.available_models)} models")
print()

stt.update_model_list()

print(f"After update: {len(stt.available_models)} models")
print()
for model in stt.available_models:
    print(f"  {model['lang']:8s} ({model['lang_text']:20s}): {model['name']}")
