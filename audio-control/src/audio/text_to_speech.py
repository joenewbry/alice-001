#!/usr/bin/env python3
"""
Text-to-speech module using OpenAI TTS API.
Converts text responses to spoken audio.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
import wave
import tempfile
from typing import Optional
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio


class TextToSpeech:
    """Converts text to speech using OpenAI TTS"""
    
    def __init__(self, voice: str = "alloy", model: str = "tts-1"):
        """
        Initialize TTS system.
        
        Args:
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model (tts-1 for faster, tts-1-hd for higher quality)
        """
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            raise ValueError("OPEN_AI_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model
        
        # Audio playback
        self.audio = pyaudio.PyAudio()
        
        # Find Waveshare/USB audio output device
        self.output_device = self._find_waveshare_device()
        
        print(f"✓ TTS initialized with voice '{voice}'")
    
    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            print("⚠️ No text to speak")
            return False
        
        try:
            print(f"🔊 Speaking: \"{text}\"")
            
            # Generate speech
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                response.stream_to_file(temp_path)
            
            # Play audio
            self._play_audio_file(temp_path, blocking=blocking)
            
            # Cleanup
            os.unlink(temp_path)
            
            return True
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False
    
    def _find_waveshare_device(self) -> Optional[int]:
        """Find Waveshare/USB audio output device"""
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                # Look for USB PnP Audio Device (Waveshare)
                if ('USB' in dev['name'] and 'PnP' in dev['name']) or 'Solid State' in dev['name']:
                    if dev['max_output_channels'] > 0:
                        print(f"✓ Found Waveshare output device: {dev['name']} (index {i})")
                        return i
            
            print("⚠️ Waveshare device not found, using default output")
            return None
        except Exception as e:
            print(f"⚠️ Error finding audio device: {e}")
            return None
    
    def _play_audio_file(self, file_path: str, blocking: bool = True):
        """Play audio file to Waveshare speakers"""
        try:
            # Load MP3 file using pydub
            audio = AudioSegment.from_mp3(file_path)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Reshape for stereo if needed
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Normalize
            samples = samples.astype(np.float32) / (2**15)
            
            # Play to specific device
            if self.output_device is not None:
                sd.play(samples, audio.frame_rate, device=self.output_device, blocking=blocking)
            else:
                sd.play(samples, audio.frame_rate, blocking=blocking)
            
            if blocking:
                sd.wait()
                
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
            # Fallback to system command
            import platform
            system = platform.system()
            if system == "Darwin":  # macOS
                os.system(f"afplay {file_path} {'&' if not blocking else ''}")
    
    def save_speech(self, text: str, output_file: str) -> bool:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"💾 Generating speech to: {output_file}")
            
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            response.stream_to_file(output_file)
            print(f"✓ Speech saved to {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ TTS save error: {e}")
            return False
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.audio.terminate()
        print("✓ TTS cleaned up")


def demo_text_to_speech():
    """Demo text-to-speech functionality"""
    import sys
    
    # Create TTS instance
    tts = TextToSpeech(voice="alloy")
    
    try:
        # Check if text provided
        if len(sys.argv) > 1:
            text = ' '.join(sys.argv[1:])
        else:
            text = "Hello! I am your robot arm assistant. I understand commands like move left, grasp, and please dance."
        
        # Speak the text
        tts.speak(text)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        tts.cleanup()


if __name__ == "__main__":
    demo_text_to_speech()

