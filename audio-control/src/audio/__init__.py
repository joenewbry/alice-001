"""Audio module for speech input/output"""

from .capture import AudioCapture
from .speech_to_text import SpeechToText
from .text_to_speech import TextToSpeech

__all__ = ['AudioCapture', 'SpeechToText', 'TextToSpeech']

