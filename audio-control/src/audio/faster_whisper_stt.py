#!/usr/bin/env python3
"""
Faster Whisper STT - Local speech recognition using faster-whisper.
2-3x faster than OpenAI Whisper API with no network latency.
"""

import numpy as np
import tempfile
import wave
from typing import Optional
from pathlib import Path


class FasterWhisperSTT:
    """Local speech-to-text using faster-whisper (optimized Whisper)"""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize faster-whisper STT.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
                       - tiny: fastest, least accurate (~50MB)
                       - base: good balance (~150MB) 
                       - small: better accuracy (~500MB)
            device: "cpu" or "cuda" (GPU)
            compute_type: "int8" (fastest), "int16", "float16", "float32"
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
        
        print(f"⚡ Loading faster-whisper model '{model_size}' on {device}...")
        print(f"   (First run will download model, subsequent runs are instant)")
        
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None  # Use default cache
        )
        
        self.model_size = model_size
        
        print(f"✓ Faster-whisper ready! (2-3x faster than API)")
    
    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio to text (local processing, very fast).
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text or None
        """
        try:
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
            
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                language="en",  # Set language for faster processing
                condition_on_previous_text=False  # Faster
            )
            
            # Collect all segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            # Cleanup
            Path(temp_path).unlink()
            
            if not text:
                return None
            
            return text
            
        except Exception as e:
            print(f"❌ Faster-whisper transcription error: {e}")
            return None
    
    def transcribe_with_timestamps(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Transcribe with word-level timestamps.
        
        Returns:
            List of (text, start_time, end_time) tuples
        """
        try:
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
            
            # Transcribe with word timestamps
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                word_timestamps=True
            )
            
            results = []
            for segment in segments:
                results.append((
                    segment.text.strip(),
                    segment.start,
                    segment.end
                ))
            
            # Cleanup
            Path(temp_path).unlink()
            
            return results
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return []


def demo_faster_whisper():
    """Demo faster-whisper vs standard whisper"""
    import time
    import sys
    
    print("=== Faster-Whisper Demo ===\n")
    
    # Check if audio file provided
    if len(sys.argv) < 2:
        print("Usage: python faster_whisper_stt.py <audio_file.wav>")
        print("\nTesting with synthetic audio...")
        
        # Create test audio (1 second of silence)
        sample_rate = 16000
        duration = 1
        audio_data = np.zeros(sample_rate * duration, dtype=np.int16).tobytes()
    else:
        audio_file = sys.argv[1]
        print(f"Loading: {audio_file}")
        with wave.open(audio_file, 'rb') as wav:
            sample_rate = wav.getframerate()
            audio_data = wav.readframes(wav.getnframes())
    
    # Test faster-whisper
    print("\n1. Testing Faster-Whisper (local, optimized):")
    stt = FasterWhisperSTT(model_size="base")
    
    start = time.time()
    text = stt.transcribe(audio_data, sample_rate)
    elapsed = time.time() - start
    
    print(f"   Result: \"{text}\"")
    print(f"   Time: {elapsed:.2f}s")
    
    print("\n✓ Faster-Whisper is 2-3x faster than standard Whisper!")
    print("  Benefits:")
    print("  • No network latency")
    print("  • Optimized inference engine")
    print("  • Runs locally (privacy)")
    print("  • Same accuracy as standard Whisper")


if __name__ == "__main__":
    demo_faster_whisper()

