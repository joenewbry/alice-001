#!/usr/bin/env python3
"""Quick microphone test"""

import sys
sys.path.insert(0, 'src')
import yaml

# Load config
with open('config/audio_config.yaml', 'r') as f:
    config = yaml.safe_load(f)['audio']

print('=' * 60)
print('MICROPHONE TEST WITH DEBUG OUTPUT')
print('=' * 60)
print('')
print('Threshold: 50 (very sensitive)')
print('This will show energy levels as you speak.')
print('')
print('🎤 SPEAK NOW! Say something like \"move left\" or \"hello robot\"')
print('   (Will listen for up to 10 seconds or until you stop speaking)')
print('')

from audio.capture import AudioCapture
capture = AudioCapture(config)

try:
    audio_data = capture.record_utterance()

    print('')
    print('=' * 60)
    if audio_data:
        duration = len(audio_data) / 2 / config['sample_rate']
        print(f'✅ SUCCESS!')
        print(f'   Captured {duration:.2f} seconds of audio')
        print(f'   Size: {len(audio_data)} bytes')
        print('')
        print('🎉 Your microphone is working!')
    else:
        print('❌ No audio captured')
        print('')
        print('Troubleshooting:')
        print('1. Check microphone is plugged in')
        print('2. Increase input volume in System Settings > Sound > Input')
        print('3. Make sure "USB PnP Audio Device" is selected')
        print('4. Speak loudly and clearly')
    print('=' * 60)

except KeyboardInterrupt:
    print('\\n\\n⏹️  Test stopped')
finally:
    capture.cleanup()

