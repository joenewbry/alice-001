# 🎤 Microphone Volume Too Low - Fix Required

## Current Issue

The Waveshare microphone is working, but the input volume is **too low**:
- **Current levels:** 20-230 (max 1322)
- **Required levels:** 500-5000+ for voice detection
- **Result:** Voice Activity Detection doesn't trigger

## Fix #1: Increase macOS Input Volume (RECOMMENDED)

1. Open **System Settings** (or System Preferences)
2. Go to **Sound** → **Input**
3. Select **"USB PnP Audio Device"** (Waveshare)
4. **Drag the "Input volume" slider to maximum** (or at least 75%)
5. Test: Speak and watch the input level meters - they should reach the middle

## Fix #2: Disable VAD (Temporary Workaround)

If adjusting volume doesn't help, we can make the system more sensitive:

Edit `config/audio_config.yaml` and change:
```yaml
vad:
  enabled: false  # Changed from true
```

This will capture all audio without voice detection.

## Fix #3: Use Energy-Based Detection (Better)

We can adjust the energy threshold to work with quieter audio.

## Test Your Microphone Volume

Run this test after adjusting volume:
```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
source venv/bin/activate
python -c "
import pyaudio
import numpy as np

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
                   input=True, input_device_index=1, frames_per_buffer=1024)

print('Speak now for 3 seconds...')
for i in range(48):
    data = stream.read(1024, exception_on_overflow=False)
    level = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
    if i % 8 == 0:
        print(f'Level: {level:.0f} {\"✅ GOOD\" if level > 500 else \"❌ TOO LOW\"}')

stream.close()
audio.terminate()
"
```

**Goal:** See "✅ GOOD" when speaking!

## Current Detected Devices

- Input: USB PnP Audio Device (index 1) ✅ Found
- Output: USB PnP Audio Device (index 0) ✅ Working

---

**Next Step:** Increase the microphone input volume in System Settings!
