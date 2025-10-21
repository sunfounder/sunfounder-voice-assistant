"""Audio playback module using PyAudio.

This module provides comprehensive audio playback functionality using the PyAudio library,
supporting both raw audio data and audio file playback with advanced buffering and gain control.
"""

import wave
import threading
import numpy as np
from typing import Optional
from .utils import redirect_error_2_null, cancel_redirect_error

# Check PyAudio availability
_pyaudio_available = False
try:
    import pyaudio
    _pyaudio_available = True
except ImportError:
    pass


class AudioPlayer:
    """Audio player for raw audio data and audio files using PyAudio.
    
    This class provides audio playback capabilities with features like:
    - Real-time audio streaming with buffering
    - Volume gain control with clipping prevention
    - Asynchronous playback support
    - Audio file playback (WAV format)
    - Audio file gain adjustment
    """

    def __init__(self,
        sample_rate: int = 22050,
        channels: int = 1,
        gain: float = 1.0,
        format: int = pyaudio.paInt16,
        timeout: Optional[float] = None,
        enable_buffering: bool = True,
        buffer_size: int = 8192) -> None:
        """Initializes the audio player with specified parameters.
        
        Args:
            sample_rate (int): Audio sample rate in Hz (default: 22050).
            channels (int): Number of audio channels (1 for mono, 2 for stereo).
            gain (float): Volume gain factor (1.0 = original volume).
            format (int): PyAudio format constant (default: pyaudio.paInt16).
            timeout (float): Timeout in seconds for playback operations.
            enable_buffering (bool): Enable audio buffering to reduce noise artifacts.
            buffer_size (int): Minimum buffer size for playback in bytes.
        
        Raises:
            ImportError: If PyAudio is not available on the system.
        """
        if not _pyaudio_available:
            raise ImportError("PyAudio is required but not available. Please install it with 'pip install pyaudio'")
            
        # Audio configuration
        self.sample_rate = sample_rate
        self.channels = channels
        self.gain = gain
        self.format = format
        self._timeout = timeout
        
        # PyAudio instance and stream management
        self._pyaudio = pyaudio.PyAudio()
        self._stream = None
        self._playback_thread = None
        self._stop_event = threading.Event()
        
        # Buffering configuration
        self._enable_buffering = enable_buffering
        self._buffer_size = buffer_size
        self._audio_buffer = bytearray()
        
        # Audio format to numpy dtype mapping for gain processing
        self._format_to_dtype = {
            pyaudio.paInt8: np.int8,
            pyaudio.paInt16: np.int16,
            pyaudio.paInt24: np.int32,  # Using int32 for 24-bit since numpy doesn't have int24
            pyaudio.paInt32: np.int32,
            pyaudio.paFloat32: np.float32
        }
        
        # Error redirection state
        self.old_stderr = None

    def __enter__(self):
        """Context manager entry point - initializes audio stream.
        
        Returns:
            AudioPlayer: The initialized audio player instance.
        """
        self.old_stderr = redirect_error_2_null()
        self._open_stream()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb) -> None:
        """Context manager exit point - cleans up audio resources.
        
        Args:
            exc_type (type): Exception type if any occurred.
            exc_val (Exception): Exception value if any occurred.
            exc_tb (traceback): Exception traceback if any occurred.
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
        """Opens the audio output stream if not already open.
        
        Creates a new PyAudio output stream with the configured parameters.
        """
        if self._stream is None or self._stream.is_stopped():
            self._stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )

    def set_gain(self, gain: float) -> None:
        """Sets the playback gain factor.
        
        Args:
            gain (float): Volume gain factor (0.0 to 2.0+). Values below 0.0 are clamped to 0.0.
        """
        self.gain = max(0.0, gain)  # Ensure gain is non-negative
    
    def get_gain(self) -> float:
        """Gets the current playback gain factor.
        
        Returns:
            float: Current gain factor (1.0 = original volume).
        """
        return self.gain
    
    def _apply_gain(self, audio_bytes: bytes) -> bytes:
        """Applies gain to audio bytes with clipping prevention.
        
        This method converts audio bytes to numpy array, applies gain with clipping
        to prevent distortion, and converts back to bytes.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to process.
        
        Returns:
            bytes: Volume-adjusted audio bytes with clipping protection.
        """
        if self.gain == 1.0:  # No gain change needed
            return audio_bytes
        
        try:
            # Get numpy dtype corresponding to audio format
            dtype = self._format_to_dtype.get(self.format, np.int16)
            
            # Ensure audio bytes length is a multiple of dtype itemsize
            itemsize = np.dtype(dtype).itemsize
            if len(audio_bytes) % itemsize != 0:
                # Truncate to nearest multiple of itemsize to avoid alignment issues
                trimmed_length = (len(audio_bytes) // itemsize) * itemsize
                if trimmed_length == 0:
                    # If trimming results in zero length, return original
                    return audio_bytes
                audio_bytes = audio_bytes[:trimmed_length]
            
            # Convert bytes to numpy array for processing
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            
            # Apply gain with clipping to prevent distortion
            audio_array = (audio_array * self.gain)
            
            # Clip values to prevent overflow and underflow
            max_val = np.iinfo(dtype).max
            min_val = np.iinfo(dtype).min
            audio_array = np.clip(audio_array, min_val, max_val)
            
            # Convert back to original dtype
            audio_array = audio_array.astype(dtype)
            
            # Convert back to bytes for playback
            return audio_array.tobytes()
        except Exception as e:
            print(f"Error applying gain: {e}")
            # Print debug information for troubleshooting
            print(f"  - Gain value: {self.gain}")
            print(f"  - Data type: {dtype}")
            print(f"  - Audio bytes length: {len(audio_bytes)}")
            return audio_bytes  # Return original if error
    
    def play(self, audio_bytes: bytes) -> None:
        """Plays audio bytes with minimal buffering for real-time streaming.
        
        This method prioritizes low-latency playback while maintaining
        audio frame integrity for true streaming characteristics.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to play.
        """
        # Reset stop event for new playback
        self._stop_event.clear()
        
        # Apply gain if needed
        audio_bytes = self._apply_gain(audio_bytes)
        
        if not self._enable_buffering:
            # Direct playback without buffering
            self._open_stream()
            
            # Frame size for 16-bit mono audio (2 bytes per sample)
            frame_size = 2  # 16-bit mono = 2 bytes per frame
            
            # Ensure data size is multiple of frame size
            if len(audio_bytes) % frame_size != 0:
                # Trim to nearest frame boundary
                play_size = (len(audio_bytes) // frame_size) * frame_size
                if play_size > 0:
                    audio_bytes = audio_bytes[:play_size]
                else:
                    return  # No valid audio data
            
            if len(audio_bytes) > 0:
                self._stream.write(audio_bytes)
            return
        
        # Minimal buffering for real-time streaming
        self._audio_buffer.extend(audio_bytes)
        
        # Frame size for 16-bit mono audio (2 bytes per sample)
        frame_size = 2  # 16-bit mono = 2 bytes per frame
        
        # Play immediately when we have enough data for smooth playback
        # Use smaller threshold for real-time responsiveness
        min_play_threshold = max(512, frame_size * 4)  # At least 4 frames
        
        while len(self._audio_buffer) >= min_play_threshold and not self._stop_event.is_set():
            # Calculate play size as multiple of frame size
            # Use smaller chunks for lower latency
            target_play_size = min(len(self._audio_buffer), 2048)  # Max 2KB per chunk
            play_size = (target_play_size // frame_size) * frame_size
            
            if play_size > 0:
                # Extract buffer data for playback
                play_data = bytes(self._audio_buffer[:play_size])
                self._audio_buffer = self._audio_buffer[play_size:]
                
                # Play the data immediately
                self._open_stream()
                self._stream.write(play_data)
    
    def flush_buffer(self) -> None:
        """Plays all remaining audio data in the buffer.
        
        When using buffering mechanism, call this method to ensure all buffered
        audio data is played out completely.
        """
        if self._enable_buffering and len(self._audio_buffer) > 0:
            # Ensure playback data size is multiple of frame size
            frame_size = 2  # Frame size for 16-bit mono audio
            play_size = (len(self._audio_buffer) // frame_size) * frame_size
            
            if play_size > 0:
                # Apply gain to remaining buffer data
                adjusted_audio = self._apply_gain(bytes(self._audio_buffer[:play_size]))
                # Play remaining buffered data
                self._stream.write(adjusted_audio)
            
            # Clear the buffer
            self._audio_buffer = bytearray()

    def play_async(self, audio_bytes: bytes) -> None:
        """Plays raw audio bytes asynchronously in a separate thread.
        
        This method allows non-blocking audio playback by running the playback
        in a background thread.
        
        Args:
            audio_bytes (bytes): Raw audio bytes to play asynchronously.
        """
        # Stop any ongoing playback
        self.stop()
        # Reset stop event for new playback
        self._stop_event.clear()
        # Start new playback thread
        self._playback_thread = threading.Thread(
            target=self.play, 
            args=(audio_bytes,)
        )
        self._playback_thread.daemon = True
        self._playback_thread.start()

    def play_file(self, file_path: str, chunk_size: int = 4096) -> None:
        """Plays an audio file (WAV format) with buffered playback.
        
        This method reads and plays WAV audio files with proper format detection
        and buffering to ensure smooth playback.
        
        Args:
            file_path (str): Path to the audio file (WAV format).
            chunk_size (int, optional): Size of audio chunks to read and play. Defaults to 4096.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported or invalid.
        """
        # Reset stop event for new playback
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
                
                # Set minimum chunk size for efficient playback
                effective_chunk_size = max(chunk_size, 4096)
                
                # Open a new stream with file parameters
                temp_stream = self._pyaudio.open(
                    format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    output=True
                )
                
                try:
                    # Use buffer to reduce playback interruptions
                    file_buffer = bytearray()
                    min_buffer_size = 8192
                    
                    # Read and play audio data
                    data = wf.readframes(effective_chunk_size)
                    while data and not self._stop_event.is_set():
                        file_buffer.extend(data)
                        
                        # Play when buffer reaches sufficient size
                        while len(file_buffer) >= min_buffer_size and not self._stop_event.is_set():
                            # Extract buffer data for playback
                            play_data = bytes(file_buffer[:min_buffer_size])
                            file_buffer = file_buffer[min_buffer_size:]
                            
                            # Apply gain if needed
                            adjusted_data = self._apply_gain(play_data)
                            temp_stream.write(adjusted_data)
                        
                        data = wf.readframes(effective_chunk_size)
                    
                    # Play remaining buffered data
                    if len(file_buffer) > 0 and not self._stop_event.is_set():
                        adjusted_data = self._apply_gain(bytes(file_buffer))
                        temp_stream.write(adjusted_data)
                finally:
                    # Clean up the temporary stream
                    temp_stream.stop_stream()
                    temp_stream.close()

                    
        except wave.Error as e:
            raise ValueError(f"Invalid WAV file: {file_path}") from e
        finally:
            cancel_redirect_error(old_stderr)

    def play_file_async(self, file_path: str, chunk_size: int = 1024) -> None:
        """Plays an audio file asynchronously in a separate thread.
        
        This method allows non-blocking audio file playback by running the file
        playback in a background thread.
        
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
        """Stops any ongoing playback and clears the audio buffer.
        
        This method signals the stop event, waits for playback threads to finish,
        and clears any buffered audio data.
        """
        # Signal stop event to terminate playback
        self._stop_event.set()
        # Wait for playback thread to finish with timeout
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=self._timeout)
        # Clear audio buffer
        self._audio_buffer = bytearray()

    def gain_file(self, input_file: str, output_file: str, gain: float) -> bool:
        """Applies gain adjustment to an audio file and saves the result.
        
        This method reads a WAV file, applies the specified gain factor with
        proper clipping protection, and writes the result to a new file.
        
        Args:
            input_file (str): Input audio file path.
            output_file (str): Output audio file path.
            gain (float): Gain factor to apply.
            
        Returns:
            bool: True if successful, False if an error occurred.
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
                    3: np.int32,  # Using int32 for 24-bit since numpy doesn't have int24
                    4: np.int32 if sample_width == 4 and wf_in.getcomptype() == 'NONE' else np.float32
                }
                
                if sample_width not in dtype_map:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                dtype = dtype_map[sample_width]
                
                # Read all audio data
                audio_data = wf_in.readframes(frames)
                
                # Convert to numpy array for processing
                audio_array = np.frombuffer(audio_data, dtype=dtype)
                
                # Apply gain with automatic type conversion
                audio_array = (audio_array * gain).astype(dtype)
                
                # Convert back to bytes
                adjusted_data = audio_array.tobytes()
                
                # Write to output file with original parameters
                with wave.open(output_file, 'wb') as wf_out:
                    wf_out.setnchannels(channels)
                    wf_out.setsampwidth(sample_width)
                    wf_out.setframerate(sample_rate)
                    wf_out.writeframes(adjusted_data)
            
            return True
        except Exception as e:
            print(f"[ERROR] gain_file error: {e}")
            return False
    
    @staticmethod
    def is_available() -> bool:
        """Checks if PyAudio is available on the system.
        
        Returns:
            bool: True if PyAudio is available, False otherwise.
        """
        return _pyaudio_available
