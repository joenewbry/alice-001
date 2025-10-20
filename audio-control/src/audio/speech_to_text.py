#!/usr/bin/env python3
"""
Speech-to-text module using OpenAI Whisper.
Converts audio recordings to text commands.
"""

import whisper
import numpy as np
import os
from typing import Optional
import tempfile
import wave


class SpeechToText:
    """Converts speech to text using Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
                       'base' is a good balance of speed and accuracy
        """
        self.model_name = model_name
        print(f"📥 Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print(f"✓ Whisper model loaded")
        
    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not audio_data:
            print("⚠️ No audio data to transcribe")
            return None
        
        try:
            # Save audio to temporary file (Whisper needs a file)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
            
            # Transcribe
            print("🔄 Transcribing audio...")
            result = self.model.transcribe(
                temp_path,
                language="en",
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=False
            )
            
            text = result['text'].strip()
            confidence = result.get('confidence', 0.0)
            
            print(f"✓ Transcribed: \"{text}\"")
            
            # Cleanup
            os.unlink(temp_path)
            
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return None
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            print(f"🔄 Transcribing file: {audio_file}")
            result = self.model.transcribe(
                audio_file,
                language="en",
                fp16=False,
                verbose=False
            )
            
            text = result['text'].strip()
            print(f"✓ Transcribed: \"{text}\"")
            
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None


def demo_speech_to_text():
    """Demo speech-to-text functionality"""
    import sys
    
    # Create STT instance
    stt = SpeechToText(model_name="base")
    
    # Check if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        text = stt.transcribe_file(audio_file)
        if text:
            print(f"\n📝 Result: {text}")
    else:
        print("Usage: python speech_to_text.py <audio_file.wav>")


if __name__ == "__main__":
    demo_speech_to_text()

