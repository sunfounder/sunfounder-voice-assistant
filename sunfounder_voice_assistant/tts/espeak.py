from .._utils import is_installed, run_command, check_executable
from .._audio_player import AudioPlayer
from .._base import _Base

class Espeak(_Base):
    """ Espeak TTS engine
    
    Args:
        *args: passed to :class:`sunfounder_voice_assistant._base._Base`.
        **kwargs: passed to :class:`sunfounder_voice_assistant._base._Base`.
    """
    ESPEAK = 'espeak'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not is_installed("espeak"):
            raise Exception("TTS engine: espeak is not installed.")
        self._amp = 100
        self._speed = 175
        self._gap = 5
        self._pitch = 50
        self._lang = 'en-US'

    def tts(self, words: str, file_path: str) -> None:
        """ Text-to-speech with espeak

        Args:
            words (str): Word to say
            file_path (str): Path to save audio file
        """
        self.log.debug(f'espeak: [{words}]')
        if not check_executable('espeak'):
            self.log.debug('espeak is busy. Pass')

        cmd = f'espeak -a{self._amp} -s{self._speed} -g{self._gap} -p{self._pitch} "{words}" -w {file_path}'
        _, result = run_command(cmd)
        if len(result) != 0:
            raise Exception(f'tts-espeak:\n\t{result}')
        self.log.debug(f'command: {cmd}')

    def say(self, words: str) -> None:
        """ Say words with espeak

        Args:
            words (str): Words to say
        """
        file = '/tmp/espeak.wav'
        self.tts(words, file)
        with AudioPlayer() as player:
            player.play_file(file)
        
    def set_amp(self, amp: int) -> None:
        """ Set amplitude

        Args:
            amp (int): Amplitude (0-200)
        """
        if not isinstance(amp, int) or not 0 <= amp <= 200:
            raise ValueError(f'Amp should be in 0 to 200, not "{amp}"')
        self._amp = amp

    def set_speed(self, speed: int) -> None:
        """ Set speed

        Args:
            speed (int): Speed (80-260)
        """
        if not isinstance(speed, int) or not 80 <= speed <= 260:
            raise ValueError(f'speed should be in 80 to 260, not "{speed}"')
        self._speed = speed

    def set_gap(self, gap: int) -> None:
        """ Set gap

        Args:
            gap (int): Gap (0-200)
        """
        if not isinstance(gap, int) or not 0 <= gap <= 200:
            raise ValueError(f'gap should be in 0 to 200, not "{gap}"')
        self._gap = gap

    def set_pitch(self, pitch: int) -> None:
        """ Set pitch

        Args:
            pitch (int): Pitch (0-99)
        """
        if not isinstance(pitch, int) or not 0 <= pitch <= 99:
            raise ValueError(f'pitch should be in 0 to 99, not "{pitch}"')
        self._pitch = pitch
