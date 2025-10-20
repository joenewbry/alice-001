#!/usr/bin/env python3
"""
Audio capture module for Waveshare USB Audio device.
Handles microphone input with Voice Activity Detection (VAD).
"""

import pyaudio
import wave
import numpy as np
import webrtcvad
from typing import Optional, Callable
import threading
import time
from collections import deque


class AudioCapture:
    """Captures audio from Waveshare USB device with VAD"""
    
    def __init__(self, config: dict):
        """
        Initialize audio capture.
        
        Args:
            config: Audio configuration dictionary from audio_config.yaml
        """
        self.config = config
        self.sample_rate = config['sample_rate']
        self.channels = config['channels']
        self.chunk_size = config['chunk_size']
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(config['vad']['aggressiveness'])
        self.vad_frame_duration = config['vad']['frame_duration_ms']
        self.vad_padding_duration = config['vad']['padding_duration_ms']
        
        # Recording settings
        self.silence_threshold = config['recording']['silence_threshold']
        self.max_duration = config['recording']['max_recording_duration']
        self.min_duration = config['recording']['min_recording_duration']
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.device_index = self._find_device()
        
        # State
        self.is_recording = False
        self.recorded_frames = []
        
        # Muting (to prevent capturing our own TTS output)
        self.is_muted = False
        self._mute_lock = threading.Lock()
        
        print(f"✓ Audio capture initialized: {self.sample_rate}Hz, device: {self.device_index}")
    
    def _find_device(self) -> Optional[int]:
        """Find Waveshare USB Audio device index"""
        device_name = self.config.get('device_name', 'Waveshare')
        
        # Try to find the device
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info['name'].lower()
            
            # Look for USB PnP Audio Device (Waveshare) with input channels
            if (('usb' in name and 'pnp' in name) or device_name.lower() in name) and info['maxInputChannels'] > 0:
                print(f"✓ Found Waveshare input device: {info['name']} (index {i})")
                return i
        
        print(f"⚠️ Waveshare device not found, using default input device")
        return None
    
    def mute(self):
        """
        Mute the microphone (prevent recording).
        Used to prevent capturing our own TTS output.
        """
        with self._mute_lock:
            self.is_muted = True
            print("🔇 Microphone muted (preventing audio feedback)")
    
    def unmute(self):
        """
        Unmute the microphone (allow recording).
        """
        with self._mute_lock:
            self.is_muted = False
            print("🔊 Microphone unmuted (ready for input)")
    
    def start_stream(self):
        """Start audio input stream"""
        if self.stream is not None:
            print("⚠️ Stream already started")
            return
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=None
        )
        
        print("🎤 Audio stream started")
    
    def stop_stream(self):
        """Stop audio input stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("🔇 Audio stream stopped")
    
    def record_utterance(self, callback: Optional[Callable] = None) -> bytes:
        """
        Record a single utterance (speech segment) with VAD.
        
        Args:
            callback: Optional callback for audio level updates
            
        Returns:
            Audio data as bytes, or None if muted
        """
        # Check if muted (don't record our own TTS output)
        with self._mute_lock:
            if self.is_muted:
                return None
        if self.stream is None:
            self.start_stream()
        
        frames = []
        num_padding_frames = int(self.vad_padding_duration / self.vad_frame_duration)
        ring_buffer = deque(maxlen=num_padding_frames)
        triggered = False
        
        voiced_frames = []
        num_unvoiced = 0
        silence_frames_needed = int(self.silence_threshold * self.sample_rate / self.chunk_size)
        
        start_time = time.time()
        
        print("🎤 Listening...")
        
        while True:
            # Check max duration
            if time.time() - start_time > self.max_duration:
                print("⏱️ Max recording duration reached")
                break
            
            # Read audio chunk
            try:
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception as e:
                print(f"❌ Error reading audio: {e}")
                break
            
            # Check if speech is present using VAD
            is_speech = self._is_speech(audio_chunk)
            
            if not triggered:
                ring_buffer.append((audio_chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                
                # Start recording when speech detected
                if num_voiced > 0.8 * ring_buffer.maxlen:
                    triggered = True
                    print("🗣️ Speech detected, recording...")
                    # Add buffered frames
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(audio_chunk)
                ring_buffer.append((audio_chunk, is_speech))
                
                # Check for end of speech
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
                if num_unvoiced > 0.8 * ring_buffer.maxlen:
                    num_unvoiced += 1
                    if num_unvoiced >= silence_frames_needed:
                        print("🔇 Silence detected, stopping recording")
                        break
            
            # Audio level callback
            if callback:
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                level = np.abs(audio_array).mean() / 32768.0
                callback(level)
        
        # Check minimum duration
        duration = len(voiced_frames) * self.chunk_size / self.sample_rate
        if duration < self.min_duration:
            print(f"⚠️ Recording too short: {duration:.2f}s")
            return b''
        
        print(f"✓ Recorded {duration:.2f}s of audio")
        return b''.join(voiced_frames)
    
    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech using VAD or energy detection"""
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.abs(audio_array).mean()
        
        # Use energy-based detection if VAD is disabled or for low-level audio
        if not self.config['vad'].get('enabled', True):
            # Very low threshold for quieter microphones (50 = almost any sound)
            threshold = 50
            is_speech = energy > threshold
            # Debug output every 20th frame
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            if self._debug_counter % 20 == 0:
                status = "🔊" if is_speech else "🔇"
                print(f"  Energy: {energy:6.1f} {status}")
            return is_speech
        
        try:
            # VAD expects 10, 20, or 30ms frames at 8, 16, 32kHz
            is_vad_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
            # Also use energy threshold as backup for low volume
            is_energy_speech = energy > 100
            return is_vad_speech or is_energy_speech
        except Exception as e:
            # If VAD fails, use energy-based detection as fallback
            return energy > 100
    
    def save_audio(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        print(f"💾 Audio saved to {filename}")
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.stop_stream()
        self.audio.terminate()
        print("✓ Audio capture cleaned up")


def demo_audio_capture():
    """Demo audio capture functionality"""
    import yaml
    
    # Load config
    with open('config/audio_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['audio']
    
    # Create capture instance
    capture = AudioCapture(config)
    
    try:
        # Record utterance
        audio_data = capture.record_utterance()
        
        if audio_data:
            # Save to file
            capture.save_audio(audio_data, 'test_recording.wav')
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        capture.cleanup()


if __name__ == "__main__":
    demo_audio_capture()

