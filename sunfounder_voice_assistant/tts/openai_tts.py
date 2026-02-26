import requests
from .._audio_player import AudioPlayer
from .._base import _Base

from typing import Optional
from enum import StrEnum

class OpenAI_TTS(_Base):
    """ OpenAI TTS engine.
    
    Args:
        voice (Voice, optional): Voice, default is Voice.ALLOY.
        model (Model, optional): Model, default is Model.GPT_4O_MINI_TTS.
        api_key (str, optional): API key.
        gain (float, optional): Volume gain, default is 1.5.
        log (logging.Logger, optional): Logger, default is None.
        *args: passed to :class:`sunfounder_voice_assistant._base._Base`.
        **kwargs: passed to :class:`sunfounder_voice_assistant._base._Base`.
    """

    DEFAULT_INSTRUCTIONS = "Speak in a cheerful and positive tone."

    URL = "https://api.openai.com/v1/audio/speech"
    AUDIO_FORMAT = 'wav'
    class Voice(StrEnum):
        """ Voice enum. """

        ALLOY = "alloy"
        ASH = "ash"
        BALLAD = "ballad"
        CORAL = "coral"
        ECHO = "echo"
        FABLE = "fable"
        NOVA = "nova"
        ONYX = "onyx"
        SAGE = "sage"
        SHIMMER = "shimmer"

    class Model(StrEnum):
        """ Model enum. """

        GPT_4O_MINI_TTS = "gpt-4o-mini-tts"

    DEFAULT_MODEL = Model.GPT_4O_MINI_TTS
    DEFAULT_VOICE = Voice.ALLOY

    def __init__(self, *args,
        voice: Voice=DEFAULT_VOICE,
        model: Model=DEFAULT_MODEL,
        api_key: str=None,
        gain: float=1.5,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._model = model or self.DEFAULT_MODEL
        self._voice = voice or self.DEFAULT_VOICE
        self._gain = gain
        self.is_ready = False

        if api_key:
            self.set_api_key(api_key)

    def tts(self, words: str, output_file: str=f"/tmp/openai_tts.{AUDIO_FORMAT}", instructions: Optional[str]=None, stream: bool=False) -> bool:
        """ Request OpenAI TTS API.

        Args:
            words (str): Words to say.
            output_file (str, optional): Output file, default is '/tmp/openai_tts.wav'.
            instructions (str, optional): Instructions, default is None.
            stream (bool, optional): Whether to stream the audio, default is False.

        Returns:
            bool: True if success, False otherwise.
        """
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self._model.value,
            "input": words,
            "voice": self._voice.value,
            "response_format": self.AUDIO_FORMAT,
        }
        
        if instructions:
            data["instructions"] = instructions
        
        try:
            response = requests.post(self.URL, json=data, headers=headers, stream=stream)
            response.raise_for_status()
                
            if stream:
                with AudioPlayer(gain=self._gain) as player:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            player.play(chunk)
            else:
                content = response.content
                with open(output_file, "wb") as f:
                    f.write(content)
                
            return True
        
        except requests.exceptions.RequestException as e:
            self.log.error(f"OpenAI TTS API request error: {e}")
            return False
        except IOError as e:
            self.log.error(f"OpenAI TTS API file operation error: {e}")
            return False

    def say(self, words: str, instructions: Optional[str]=None, stream: bool=True) -> None:
        """ Say words.

        Args:
            words (str): Words to say.
            instructions (str, optional): Instructions, default is None.
            stream (bool, optional): Whether to stream the audio, default is True.
        """
        if stream:
            self.tts(words, instructions=instructions, stream=True)
        else:
            file_name = f"/tmp/openai_tts.{self.AUDIO_FORMAT}"
            self.tts(words, instructions=instructions, output_file=file_name, stream=False)
            with AudioPlayer(gain=self._gain) as player:
                player.play_file(file_name)

    def set_voice(self, voice: [Voice, str]) -> None:
        """ Set voice.

        Args:
            voice (Voice | str): Voice.
        """
        if isinstance(voice, str):
            voice = self.Voice(voice)
        elif not isinstance(voice, self.Voice):
            raise ValueError(f"Invalid voice: {voice}, must be {self.Voice.__name__}")
        
        self._voice = voice

    def set_model(self, model: [Model, str]) -> None:
        """ Set model.

        Args:
            model (Model | str): Model.
        """
        if isinstance(model, str):
            model = self.Model(model)
        elif not isinstance(model, self.Model):
            raise ValueError(f"Invalid model: {model}, must be {self.Model.__name__}")
        self._model = model

    def set_api_key(self, api_key: str) -> None:
        """ Set api key.

        Args:
            api_key (str): API key.
        """
        if not isinstance(api_key, str):
            raise ValueError(f"Invalid api_key: {api_key}, must be str")
        self._api_key = api_key

    def set_gain(self, gain: float) -> None:
        """ Set gain.

        Args:
            gain (float): Gain.
        """
        if not isinstance(gain, float):
            raise ValueError(f"Invalid gain: {gain}, must be float")
        self._gain = gain
