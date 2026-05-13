import json
import os
import wave
from pathlib import Path
from typing import Callable, List
from urllib.error import URLError
from urllib.request import urlopen
from piper import PiperVoice
from piper.download_voices import _needs_download, VOICE_PATTERN, URL_FORMAT
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf
from tqdm import tqdm

from .piper_models import PIPER_MODELS as _DEFAULT_PIPER_MODELS, MODELS as _DEFAULT_MODELS, COUNTRYS as _DEFAULT_COUNTRYS
from .._audio_player import AudioPlayer
from .._base import _Base

HOME = os.path.expanduser("~")
PIPER_MODEL_DIR = f"{HOME}/.piper_models"
VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json?download=true"
PIPER_MODEL_LIST_CACHE_PATH = Path(PIPER_MODEL_DIR, "voices.json")

def _parse_voices_json(voices_dict: dict) -> dict:
    """Parse HuggingFace voices.json format into PIPER_MODELS format.

    Input: {"en_US-lessac-medium": {...}, "en_US-lessac-low": {...}, ...}
    Output: {"en_US": {"lessac": ["en_US-lessac-low", "en_US-lessac-medium"], ...}, ...}
    """
    models = {}
    for voice_id in voices_dict:
        match = VOICE_PATTERN.match(voice_id)
        if not match:
            continue
        lang_code = match.group("lang_family") + "_" + match.group("lang_region")
        voice_name = match.group("voice_name")
        if lang_code not in models:
            models[lang_code] = {}
        if voice_name not in models[lang_code]:
            models[lang_code][voice_name] = []
        models[lang_code][voice_name].append(voice_id)
    return models


class Piper(_Base):
    """ Piper TTS engine.
    
    Args:
        model (str, optional): model, leave it None to use default model, defaults to None
        *args: passed to :class:`sunfounder_voice_assistant._base._Base`.
        **kwargs: passed to :class:`sunfounder_voice_assistant._base._Base`.
    """

    def __init__(self, *args, model: str = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Init model directory
        if not os.path.exists(PIPER_MODEL_DIR):
            os.makedirs(PIPER_MODEL_DIR, 0o777)
            os.chown(PIPER_MODEL_DIR, 1000, 1000)
        self._models = list(_DEFAULT_MODELS)
        self._piper_models = {k: dict(v) for k, v in _DEFAULT_PIPER_MODELS.items()}
        self._countrys = list(_DEFAULT_COUNTRYS)
        self._load_model_list()
        self.model = None
        if model is not None:
            self.set_model(model)
        else:
            self.piper = None

    def _load_model_list(self):
        """Load model list from local cache or built-in defaults (offline, no network)."""
        models = None
        piper_models = None

        if PIPER_MODEL_LIST_CACHE_PATH.exists():
            try:
                with open(PIPER_MODEL_LIST_CACHE_PATH, "r", encoding="utf-8") as f:
                    voices_dict = json.load(f)
                piper_models = _parse_voices_json(voices_dict)
                models = []
                for voices in piper_models.values():
                    for model_list in voices.values():
                        models.extend(model_list)
            except Exception:
                pass

        if not models:
            piper_models = {k: dict(v) for k, v in _DEFAULT_PIPER_MODELS.items()}
            models = list(_DEFAULT_MODELS)

        self._piper_models = piper_models
        self._models = models
        self._countrys = list(piper_models.keys())

    def update_model_list(self):
        """Fetch latest model list from network and save to cache.

        Call this manually when you want to check for new models online.
        Falls back to local cache if network is unavailable.
        """
        models = None
        piper_models = None

        try:
            with urlopen(VOICES_JSON_URL, timeout=5) as response:
                voices_dict = json.load(response)
            piper_models = _parse_voices_json(voices_dict)
            models = []
            for voices in piper_models.values():
                for model_list in voices.values():
                    models.extend(model_list)
            self.log.info(f"Model list updated from network ({len(models)} models)")

            if models:
                try:
                    with open(PIPER_MODEL_LIST_CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(voices_dict, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        except Exception:
            self.log.warning("Network unavailable, using local model list...")
            if PIPER_MODEL_LIST_CACHE_PATH.exists():
                try:
                    with open(PIPER_MODEL_LIST_CACHE_PATH, "r", encoding="utf-8") as f:
                        voices_dict = json.load(f)
                    piper_models = _parse_voices_json(voices_dict)
                    models = []
                    for voices in piper_models.values():
                        for model_list in voices.values():
                            models.extend(model_list)
                    self.log.info(f"Model list loaded from cache ({len(models)} models)")
                except Exception:
                    pass

        if not models:
            self.log.warning("No local model list available, keeping current list")

        if models:
            self._piper_models = piper_models
            self._models = models
            self._countrys = list(piper_models.keys())

    def get_language(self) -> str:
        """ Get language from model.

        Returns:
            str: language
        """
        language = self.model.split("-")[0]
        return language

    def is_model_downloaded(self, model: str) -> bool:
        """ Check if model is downloaded.

        Args:
            model (str): model

        Returns:
            bool: True if model is downloaded, False otherwise
        """
        if model is None:
            model = self.model
        onnx_file = os.path.join(PIPER_MODEL_DIR, model + ".onnx")
        json_file = onnx_file + ".json"
        onnx_exists = os.path.exists(onnx_file)
        json_exists = os.path.exists(json_file)
        self.log.debug(f"Model {model} onnx file exists: {onnx_exists}")
        self.log.debug(f"Model {model} json file exists: {json_exists}")
        return onnx_exists and json_exists
    
    def download_model(self,
                        model: str,
                        force: bool = False,
                        progress_callback: Callable[[int, int], None] = None) -> None:
        """ Download model.

        Args:
            model (str): model
            force (bool, optional): force download, default is False
            progress_callback (Callable[[int, int], None], optional): progress callback, default is None
        """ 
        model_path = self.get_model_path(model)
        if not self.is_model_downloaded(model) or force:
            self.log.info(f"Downloading model {model} to {model_path}")
            download_voice(model,
                            Path(PIPER_MODEL_DIR),
                            force_redownload=force,
                            progress_callback=progress_callback)

    def fix_chinese_punctuation(self, text: str) -> str:
        """Replace Chinese punctuation with English punctuation.

        Args:
            text (str): text

        Returns:
            str: text with English punctuation
        """
        if self.get_language() != "zh_CN":
            return text
        MAP = {
            '，': '. ',
            '。': '. ',
            '！': '! ',
            '？': '? ',
            '——': '. ',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            "~": ". ",
            "～": ". ",
            "：": ". ",
            "...": ". ",
            "……": ". ",
            "、": ". ",
        }
        for k, v in MAP.items():
            text = text.replace(k, v)
        # find number followed by dot and replace with number followed by 点
        import re
        text = re.sub(r'(\d)\.(\d)', r'\1点\2', text)

        return text

    def tts(self, text: str, file: str) -> None:
        """ Synthesize text to wave file.

        Args:
            text (str): text
            file (str): wave file path

        Raises:
            ValueError: Model not set, set model first, with Piper.set_model(model)
        """
        if self.piper is None:
            raise ValueError("Model not set, set model first, with Piper.set_model(model)")
        text = self.fix_chinese_punctuation(text)

        with wave.open(file, "wb") as wav_file:
            self.piper.synthesize_wav(text, wav_file)

    def stream(self, text: str) -> None:
        """ Stream text to speaker.

        Args:
            text (str): text

        Raises:
            ValueError: Model not set, set model first, with Piper.set_model(model)
        """
        if self.piper is None:
            raise ValueError("Model not set, set model first, with Piper.set_model(model)")
        text = self.fix_chinese_punctuation(text)

        with AudioPlayer(self.piper.config.sample_rate) as player:
            for chunk in self.piper.synthesize(text):
                player.play(chunk.audio_int16_bytes)

    def say(self, text: str, stream: bool = True) -> None:
        """ Say text.

        Args:
            text (str): text
            stream (bool, optional): stream to speaker, default is True

        Raises:
            ValueError: Model not set, set model first, with Piper.set_model(model)
        """
        if self.piper is None:
            raise ValueError("Model not set, set model first, with Piper.set_model(model)")
        if stream:
            self.stream(text)
        else:
            file = "./tts_piper.wav"
            self.tts(text, file)
            # Use the correct sample rate from Piper config
            with AudioPlayer(self.piper.config.sample_rate) as player:
                player.play_file(file)

    def available_models(self, country: str = None) -> List[str]:
        """ Get available models.

        Args:
            country (str, optional): country, leave it None to get all models, defaults to None

        Returns:
            List[str]: available models
        """
        if country is None:
            return self._models
        else:
            return self._piper_models.get(country, [])

    def available_countrys(self) -> List[str]:
        """ Get available countrys.

        Returns:
            List[str]: available countrys
        """
        return self._countrys

    def get_model_path(self, model: str) -> str:
        """ Get model path.

        Args:
            model (str): model

        Returns:
            str: model path
        """
        return os.path.join(PIPER_MODEL_DIR, model + ".onnx")

    def set_model(self, model: str) -> None:
        """ Set model.

        Args:
            model (str): model

        Raises:
            ValueError: Model not found
        """
        if model in self._models:
            model_path = self.get_model_path(model)
            if not self.is_model_downloaded(model):
                self.log.warning(f"Model {model} not downloaded, downloading...")
                self.download_model(model)
            try:
                self.piper = PiperVoice.load(model_path)
            except InvalidProtobuf as e:
                self.log.warning(f"Failed to load model {model_path}: {e}, try to redownload model.")
                self.download_model(model, force=True)
                self.piper = PiperVoice.load(model_path)
            self.model = model
        else:
            raise ValueError("Model not found")
    
def download_voice(voice: str,
                    download_dir: Path,
                    force_redownload: bool = False,
                    progress_callback: Callable[[int, int], None] = None
                    ) -> None:
    """Download a voice model and config file to a directory.

    Args:
        voice (str): voice
        download_dir (Path): download directory
        force_redownload (bool, optional): force redownload, default is False
        progress_callback (Callable[[int, int], None], optional): progress callback, default is None
    """
    voice = voice.strip()
    voice_match = VOICE_PATTERN.match(voice)
    if not voice_match:
        raise ValueError(
            f"Voice '{voice}' did not match pattern: <language>-<name>-<quality> like 'en_US-lessac-medium'",
        )

    lang_family = voice_match.group("lang_family")
    lang_code = lang_family + "_" + voice_match.group("lang_region")
    voice_name = voice_match.group("voice_name")
    voice_quality = voice_match.group("voice_quality")

    voice_code = f"{lang_code}-{voice_name}-{voice_quality}"
    format_args = {
        "lang_family": lang_family,
        "lang_code": lang_code,
        "voice_name": voice_name,
        "voice_quality": voice_quality,
    }

    # 下载模型文件（带进度条）
    model_path = download_dir / f"{voice_code}.onnx"
    if force_redownload or _needs_download(model_path):
        model_url = URL_FORMAT.format(extension=".onnx",** format_args)
        _download_with_progress(model_url, model_path, progress_callback)

    # 下载配置文件（带进度条）
    config_path = download_dir / f"{voice_code}.onnx.json"
    if force_redownload or _needs_download(config_path):
        config_url = URL_FORMAT.format(extension=".onnx.json", **format_args)
        _download_with_progress(config_url, config_path, progress_callback)

    # _LOGGER.info("Downloaded: %s", voice)


def _download_with_progress(url: str,
                            output_path: Path,
                            progress_callback: Callable[[int, int], None] = None) -> None:
    """ Download file with progress bar.

    Args:
        url (str): URL
        output_path (Path): output path
        progress_callback (Callable[[int, int], None], optional): progress callback function, default is None
    """
    with urlopen(url) as response:
        file_size = int(response.headers.get("Content-Length", 0))
        
        if progress_callback:
            progress_callback(0, file_size)
        else:
            progress_bar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {output_path.name}",
                leave=True
            )
        
        with open(output_path, "wb") as out_file:
            while True:
                chunk = response.read(8192)  # 8KB 块
                if not chunk:
                    break
                out_file.write(chunk)
                if progress_callback:
                    progress_callback(len(chunk), file_size)
                else:
                    progress_bar.update(len(chunk))
        
        if progress_callback:
            progress_callback(file_size, file_size)
        else:
            progress_bar.close()
