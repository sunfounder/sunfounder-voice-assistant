""" Audio playback using PyAudio.
This module provides audio playback functionality using PyAudio library,
supporting both raw audio data and audio file playback.
"""

import wave
import threading
import numpy as np
from typing import Optional
from .utils import redirect_error_2_null, cancel_redirect_error, ignore_stderr

# Try to import PyAudio
_pyaudio_available = False
try:
    import pyaudio
    _pyaudio_available = True
except ImportError:
    pass


class AudioPlayer:
    """ Plays raw audio and audio files using PyAudio. """

    def __init__(self,
        sample_rate: int = 22050,
        channels: int = 1,
        gain: float = 1.0,
        format: int = pyaudio.paInt16,
        timeout: Optional[float] = None) -> None:
        """ Initializes audio player.
        
        Args:
            sample_rate (int): Sample rate of audio in Hz.
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            gain (float): Volume gain factor (default: 1.0).
            format (int): PyAudio format constant (default: pyaudio.paInt16).
            timeout (float): Timeout in seconds for playback (default: None).
        
        Raises:
            ImportError: If PyAudio is not available.
        """
        if not _pyaudio_available:
            raise ImportError("PyAudio is required but not available. Please install it with 'pip install pyaudio'")
            
        self.sample_rate = sample_rate
        self.channels = channels
        self.gain = gain
        self.format = format
        self._timeout = timeout
        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self._playback_thread = None
        self._stop_event = threading.Event()
        
        # Format to numpy dtype mapping
        self._format_to_dtype = {
            pyaudio.paInt8: np.int8,
            pyaudio.paInt16: np.int16,
            pyaudio.paInt24: np.int32,  # Using int32 for 24-bit since numpy doesn't have int24
            pyaudio.paInt32: np.int32,
            pyaudio.paFloat32: np.float32
        }
        
        self.old_stderr = None

    def __enter__(self):
        """ Initializes audio stream and returns player. """
        self.old_stderr = redirect_error_2_null()
        self._open_stream()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb) -> None:
        """ Cleans up audio resources.
        
        Args:
            exc_type (type): Exception type.
            exc_val (Exception): Exception value.
            exc_tb (traceback): Exception traceback.
        """
        self.stop()
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
                pass
        try:
            self._pyaudio.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
            pass
        finally:
            if self.old_stderr is not None:
                cancel_redirect_error(self.old_stderr)
                self.old_stderr = None

    def _open_stream(self):
        """ Opens the audio stream. """
        if self._stream is None or self._stream.is_stopped():
            self._stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )

    def set_gain(self, gain: float) -> None:
        """ Sets the playback gain.
        
        Args:
            gain (float): Volume gain factor (0.0 to 2.0+). 1.0 is original gain.
        """
        self.gain = max(0.0, gain)  # Ensure gain is non-negative
    
    def get_gain(self) -> float:
        """ Gets the current playback gain.
        
        Returns:
            float: Current gain factor.
        """
        return self.gain
    
    def _apply_gain(self, audio_bytes: bytes, dtype=np.int16) -> bytes:
        """ Applies gain to audio bytes.
        
        Args:
            audio_bytes (bytes): Raw audio bytes.
            dtype (np.dtype): NumPy data type of audio samples.
        
        Returns:
            bytes: Volume-adjusted audio bytes.
        """
        if self.gain == 1.0:  # No gain change needed
            return audio_bytes
            
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            
            # Apply gain
            audio_array = (audio_array * self.gain).astype(dtype)
            
            # Convert back to bytes
            return audio_array.tobytes()
        except Exception as e:
            print(f"Error applying gain: {e}")
            return audio_bytes  # Return original if error
    
    def play(self, audio_bytes: bytes) -> None:
        """ Plays raw audio bytes.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to play.
        """
        self._open_stream()
        # Reset stop event
        self._stop_event.clear()
        
        # Apply gain if needed
        dtype = self._format_to_dtype.get(self.format, np.int16)
        adjusted_audio = self._apply_gain(audio_bytes, dtype)
        
        # Play audio
        self._stream.write(adjusted_audio)

    def play_async(self, audio_bytes: bytes) -> None:
        """ Plays raw audio bytes asynchronously in a separate thread.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to play.
        """
        # Stop any ongoing playback
        self.stop()
        # Reset stop event
        self._stop_event.clear()
        # Start new playback thread
        self._playback_thread = threading.Thread(
            target=self.play, 
            args=(audio_bytes,)
        )
        self._playback_thread.daemon = True
        self._playback_thread.start()

    def play_file(self, file_path: str, chunk_size: int = 1024) -> None:
        """ Plays an audio file.
        
        Args:
            file_path (str): Path to the audio file (WAV format).
            chunk_size (int, optional): Size of audio chunks to read and play. Defaults to 1024.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        # Reset stop event
        self._stop_event.clear()
        old_stderr = redirect_error_2_null()
        
        try:
            # Open the WAV file
            with wave.open(file_path, 'rb') as wf:
                # Get file parameters
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                
                # Map sample width to PyAudio format
                format_map = {
                    1: pyaudio.paInt8,
                    2: pyaudio.paInt16,
                    3: pyaudio.paInt24,
                    4: pyaudio.paInt32
                }
                
                if sample_width not in format_map:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                audio_format = format_map[sample_width]
                
                # Open a new stream with file parameters
                temp_stream = self._pyaudio.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    output=True
                )
                
                try:
                    # Read and play audio data
                    data = wf.readframes(chunk_size)
                    while data and not self._stop_event.is_set():
                        # Apply gain if needed
                        dtype = self._format_to_dtype.get(audio_format, np.int16)
                        adjusted_data = self._apply_gain(data, dtype)
                        temp_stream.write(adjusted_data)
                        data = wf.readframes(chunk_size)
                finally:
                    # Clean up the temporary stream
                    temp_stream.stop_stream()
                    temp_stream.close()

                    
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {file_path}") from e
        finally:
            cancel_redirect_error(old_stderr)

    def play_file_async(self, file_path: str, chunk_size: int = 1024) -> None:
        """ Plays an audio file asynchronously in a separate thread.
        
        Args:
            file_path (str): Path to the audio file (WAV format).
            chunk_size (int, optional): Size of audio chunks to read and play. Defaults to 1024.
        """
        # Stop any ongoing playback
        self.stop()
        # Start new playback thread
        self._playback_thread = threading.Thread(
            target=self.play_file, 
            args=(file_path, chunk_size)
        )
        self._playback_thread.daemon = True
        self._playback_thread.start()

    def stop(self) -> None:
        """ Stops any ongoing playback. """
        # Signal stop event
        self._stop_event.set()
        # Wait for playback thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=self._timeout)

    def gain_file(self, input_file: str, output_file: str, gain: float) -> bool:
        """ Applies gain to an audio file.
        
        Args:
            input_file (str): Input audio file path.
            output_file (str): Output audio file path.
            gain (float): Gain factor.
            
        Returns:
            bool: True if success, False otherwise.
        """
        try:
            # Open the input WAV file
            with wave.open(input_file, 'rb') as wf_in:
                # Get file parameters
                channels = wf_in.getnchannels()
                sample_rate = wf_in.getframerate()
                sample_width = wf_in.getsampwidth()
                frames = wf_in.getnframes()
                
                # Map sample width to numpy dtype
                dtype_map = {
                    1: np.int8,
                    2: np.int16,
                    3: np.int32,  # Using int32 for 24-bit
                    4: np.int32 if sample_width == 4 and wf_in.getcomptype() == 'NONE' else np.float32
                }
                
                if sample_width not in dtype_map:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                dtype = dtype_map[sample_width]
                
                # Read all audio data
                audio_data = wf_in.readframes(frames)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=dtype)
                
                # Apply gain
                audio_array = (audio_array * gain).astype(dtype)
                
                # Convert back to bytes
                adjusted_data = audio_array.tobytes()
                
                # Write to output file
                with wave.open(output_file, 'wb') as wf_out:
                    wf_out.setnchannels(channels)
                    wf_out.setsampwidth(sample_width)
                    wf_out.setframerate(sample_rate)
                    wf_out.writeframes(adjusted_data)
            
            return True
        except Exception as e:
            print(f"[ERROR] gain_file err: {e}")
            return False
    
    @staticmethod
    def is_available() -> bool:
        """ Returns true if PyAudio is available.
        
        Returns:
            Bool, True if PyAudio is available, False otherwise.
        """
        return _pyaudio_available
