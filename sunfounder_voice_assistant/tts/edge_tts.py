"""EdgeTTS — free cloud TTS using Microsoft Edge's Read Aloud service.

No API key required. Supports 100+ voices across 50+ languages.
Powered by the ``edge-tts`` Python package.

Usage::

    from sunfounder_voice_assistant.tts import EdgeTTS
    tts = EdgeTTS(voice="en-US-AriaNeural")
    tts.say("Hello world")
"""

import asyncio
import edge_tts
import subprocess

from .._audio_player import AudioPlayer
from .._base import _Base

# supports 20 languages
VOICES = {
    # English
    "en-US-AriaNeural":     "US female, natural",
    "en-US-AnaNeural":      "US female, child",
    "en-US-ChristopherNeural": "US male, warm",
    "en-US-EricNeural":     "US male, professional",
    "en-US-GuyNeural":      "US male, friendly",
    "en-US-JennyNeural":    "US female, cheerful",
    "en-US-MichelleNeural": "US female, confident",
    "en-US-RogerNeural":    "US male, authoritative",
    "en-US-SteffanNeural":  "US male, casual",
    "en-GB-RyanNeural":     "UK male, natural",
    "en-GB-SoniaNeural":    "UK female, natural",
    "en-GB-LibbyNeural":    "UK female, friendly",
    "en-GB-MaisieNeural":   "UK female, warm",
    "en-AU-NatashaNeural":  "AU female, natural",
    "en-AU-WilliamNeural":  "AU male, natural",
    "en-IN-NeerjaNeural":   "IN female, warm",
    "en-IN-PrabhatNeural":  "IN male, natural",
    # Chinese
    "zh-CN-XiaoxiaoNeural": "CN female, natural",
    "zh-CN-YunxiNeural":    "CN male, professional",
    "zh-CN-YunyangNeural":  "CN male, news style",
    "zh-CN-XiaoyiNeural":   "CN female, lively",
    "zh-CN-YunjianNeural":  "CN male, senior",
    "zh-CN-XiaochenNeural": "CN female, warm",
    "zh-CN-XiaohanNeural":  "CN female, casual",
    "zh-CN-XiaomengNeural": "CN female, young",
    "zh-TW-HsiaoChenNeural":"TW female, natural",
    "zh-TW-YunJheNeural":   "TW male, natural",
    "zh-HK-HiuMaanNeural":  "HK female, natural",
    "zh-HK-WanLungNeural":  "HK male, natural",
    # Japanese
    "ja-JP-NanamiNeural":   "JP female, natural",
    "ja-JP-KeitaNeural":    "JP male, natural",
    # Korean
    "ko-KR-SunHiNeural":    "KR female, natural",
    "ko-KR-InJoonNeural":   "KR male, natural",
    # French
    "fr-FR-DeniseNeural":   "FR female, natural",
    "fr-FR-HenriNeural":    "FR male, natural",
    "fr-CA-SylvieNeural":   "CA female, natural",
    "fr-CA-JeanNeural":     "CA male, natural",
    # German
    "de-DE-KatjaNeural":    "DE female, natural",
    "de-DE-ConradNeural":   "DE male, natural",
    # Spanish
    "es-ES-ElviraNeural":   "ES female, natural",
    "es-ES-AlvaroNeural":   "ES male, natural",
    "es-MX-DaliaNeural":    "MX female, natural",
    "es-MX-JorgeNeural":    "MX male, natural",
    # Italian
    "it-IT-ElsaNeural":     "IT female, natural",
    "it-IT-IsabellaNeural": "IT female, warm",
    "it-IT-DiegoNeural":    "IT male, natural",
    # Portuguese
    "pt-BR-FranciscaNeural":"BR female, natural",
    "pt-BR-AntonioNeural":  "BR male, natural",
    "pt-PT-RaquelNeural":   "PT female, natural",
    "pt-PT-DuarteNeural":   "PT male, natural",
    # Russian
    "ru-RU-SvetlanaNeural": "RU female, natural",
    "ru-RU-DmitryNeural":   "RU male, natural",
    # Arabic
    "ar-SA-ZariyahNeural":  "SA female, natural",
    "ar-SA-HamedNeural":    "SA male, natural",
    # Hindi
    "hi-IN-SwaraNeural":    "IN female, natural",
    "hi-IN-MadhurNeural":   "IN male, natural",
    # Thai
    "th-TH-PremwadeeNeural":"TH female, natural",
    "th-TH-NiwatNeural":    "TH male, natural",
    # Vietnamese
    "vi-VN-HoaiMyNeural":   "VN female, natural",
    "vi-VN-NamMinhNeural":  "VN male, natural",
    # Indonesian
    "id-ID-GadisNeural":    "ID female, natural",
    "id-ID-ArdiNeural":     "ID male, natural",
    # Dutch
    "nl-NL-FennaNeural":    "NL female, natural",
    "nl-NL-MaartenNeural":  "NL male, natural",
    # Polish
    "pl-PL-AgnieszkaNeural":"PL female, natural",
    "pl-PL-MarekNeural":    "PL male, natural",
    # Turkish
    "tr-TR-EmelNeural":     "TR female, natural",
    "tr-TR-AhmetNeural":    "TR male, natural",
    # Swedish
    "sv-SE-SofieNeural":    "SE female, natural",
    "sv-SE-MattiasNeural":  "SE male, natural",
    # Malay
    "ms-MY-YasminNeural":   "MY female, natural",
    "ms-MY-OsmanNeural":    "MY male, natural",
    # Filipino
    "fil-PH-BlessicaNeural":"PH female, natural",
    "fil-PH-AngeloNeural":  "PH male, natural",
}


class EdgeTTS(_Base):
    """Edge TTS engine — free cloud TTS via Microsoft Edge.

    Args:
        voice (str): Voice short name, default ``en-US-AriaNeural``.
        gain (float): Volume gain factor, default 1.0.
        *args, **kwargs: passed to :class:`_Base`.
    """

    def __init__(self, *args, voice="en-US-AriaNeural", gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._voice = voice
        self._gain = gain

    # ---- public API ----

    def tts(self, text: str, file_path: str) -> None:
        """Synthesize text to an audio file via Microsoft Edge TTS.

        Args:
            text: Text to speak.
            file_path: Output audio file path (MP3 format).

        Raises:
            edge_tts.exceptions.EdgeTTSException: If the Edge TTS API
                returns an error (e.g. invalid voice, rate-limited).
        """
        self.log.debug(f"edge-tts [{self._voice}]: {text[:80]}")
        loop = asyncio.new_event_loop()
        try:
            communicate = edge_tts.Communicate(text, self._voice)
            loop.run_until_complete(communicate.save(file_path))
        finally:
            loop.close()

    def say(self, text: str) -> None:
        """Synthesize text and play it immediately.

        Args:
            text: Text to speak.
        """
        file = "./edge_tts.mp3"
        self.tts(text, file)
        wav = "./edge_tts.wav"
        subprocess.run(["sox", file, wav], capture_output=True)
        with AudioPlayer(gain=self._gain) as player:
            player.play_file(wav)

    def set_voice(self, voice: str) -> None:
        """Set the voice by its short name (e.g. ``en-US-AriaNeural``).

        Args:
            voice: Edge TTS voice short name.
        """
        self._voice = voice

    def set_gain(self, gain: float) -> None:
        """Set volume gain.

        Args:
            gain: Volume gain factor (1.0 = original).
        """
        self._gain = gain

    @staticmethod
    def available_voices() -> dict:
        """Return a dict of commonly used ``{voice_id: description}`` pairs."""
        return dict(VOICES)
