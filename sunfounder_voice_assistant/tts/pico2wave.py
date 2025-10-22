from .._utils import is_installed, run_command, check_executable
from .._base import _Base

class Pico2Wave(_Base):
    """ Pico2Wave TTS engine.
    
    Args:
        lang (str, optional): language, leave it None to use default language, defaults to 'en-US'
        *args: passed to :class:`sunfounder_voice_assistant._base._Base`.
        **kwargs: passed to :class:`sunfounder_voice_assistant._base._Base`.
    """

    SUPPORTED_LANGUAUE = [
        'en-US',
        'en-GB',
        'de-DE',
        'es-ES',
        'fr-FR',
        'it-IT',
    ]
    """Supported languages."""

    def __init__(self, *args, lang: str=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not is_installed("pico2wave"):
            raise Exception("TTS engine: pico2wave is not installed.")
        
        self._lang = lang or 'en-US'

    def say(self, words: str) -> None:
        """ Say words with pico2wave.

        Args:
            words (str): words to say.
        """
        self.log.debug(f'pico2wave: [{words}]')
        if not check_executable('pico2wave'):
            self.log.debug('pico2wave is busy. Pass')

        cmd = f'pico2wave -l {self._lang} -w /tmp/pico2wave.wav "{words}" && aplay /tmp/pico2wave.wav 2>/dev/null & '
        _, result = run_command(cmd)
        if len(result) != 0:
            raise (f'tts-pico2wave:\n\t{result}')
        self.log.debug(f'command: {cmd}')

    def set_lang(self, lang: str) -> None:
        """ Set language.

        Args:
            lang (str): language.
        """
        if lang not in self.SUPPORTED_LANGUAUE:
            raise ValueError(f'Language {lang} is not supported')
        self._lang = lang
