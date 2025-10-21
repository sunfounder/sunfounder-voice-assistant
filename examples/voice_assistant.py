from fusion_hat.voice_assistant import VoiceAssistant
from pidog.llm import OpenAI as LLM
from secret import OPENAI_API_KEY as API_KEY

llm = LLM(
    api_key=API_KEY,
    model="gpt-4o-mini",
)

# Robot name
NAME = "Buddy"

# Enable image, need to set up a multimodal language model
WITH_IMAGE = True

# Set models and languages
LLM_MODEL = "gpt-4o-mini"
TTS_MODEL = "en_US-ryan-low"
STT_LANGUAGE = "en-us"

# Enable keyboard input
KEYBOARD_ENABLE = True

# Enable wake word
WAKE_ENABLE = True
WAKE_WORD = [f"hey {NAME.lower()}"]
# Set wake word answer, set empty to disable
ANSWER_ON_WAKE = "Hi there"

# Welcome message
WELCOME = f"Hi, I'm {NAME}. Wake me up with: " + ", ".join(WAKE_WORD)

# Set instructions
INSTRUCTIONS = f"""
You are a helpful assistant, named {NAME}.
"""

va = VoiceAssistant(
    llm,
    name=NAME,
    with_image=WITH_IMAGE,
    tts_model=TTS_MODEL,
    stt_language=STT_LANGUAGE,
    keyboard_enable=KEYBOARD_ENABLE,
    wake_enable=WAKE_ENABLE,
    wake_word=WAKE_WORD,
    answer_on_wake=ANSWER_ON_WAKE,
    welcome=WELCOME,
    instructions=INSTRUCTIONS,
)

if __name__ == "__main__":
    va.run()
