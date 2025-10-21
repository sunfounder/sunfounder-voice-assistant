#!/usr/bin/env python3

from .espeak import Espeak
from .pico2wave import Pico2Wave
from .piper import Piper
from .openai_tts import OpenAI_TTS

from enum import StrEnum

class TTS_ENGINE(StrEnum):
    PIPER = "piper"
    PICO2WAVE = "pico2wave"
    ESPEAK = "espeak"
    OPENAI = "openai"

class TTS:
    def __init__(self, engine: TTS_ENGINE = TTS_ENGINE.PIPER, *args, **kwargs) -> None:
        """ Initilize tts

        Args:
            engine (str): TTS engine name.
        """
        self.set_engine(engine, *args, **kwargs)

    def set_engine(self, engine: TTS_ENGINE, *args, **kwargs) -> None:
        """ Set the TTS engine.

        Args:
            engine (TTS_ENGINE): TTS engine name.
        """
        if engine == TTS_ENGINE.PIPER:
            self.tts = Piper(*args, **kwargs)
        elif engine == TTS_ENGINE.PICO2WAVE:
            self.tts = Pico2Wave(*args, **kwargs)
        elif engine == TTS_ENGINE.ESPEAK:
            self.tts = Espeak(*args, **kwargs)
        elif engine == TTS_ENGINE.OPENAI:
            self.tts = OpenAITTS(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")

    def __getattr__(self, name):
        return getattr(self.tts, name)

    def __setattr__(self, name, value):
        setattr(self.tts, name, value)
