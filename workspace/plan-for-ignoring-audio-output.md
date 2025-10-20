# Plan: Preventing Audio Output from Being Captured as Input

## Problem Statement

The robot's speech output (TTS audio) is being picked up by the microphone and processed as a new voice command, creating a feedback loop.

**Current Flow:**
```
User speaks → Microphone captures → Process → TTS plays → 🔴 Microphone picks up TTS → Process again (BAD)
```

**Desired Flow:**
```
User speaks → Microphone captures → Process → TTS plays → ⛔ Microphone ignores TTS → Wait for real user input
```

---

## Root Causes

1. **Same audio device** - Waveshare USB stick has both microphone and speakers
2. **Physical proximity** - Mic and speakers are close together
3. **No echo cancellation** - System doesn't filter out its own voice
4. **Continuous listening** - Mic is always active, even during speech output
5. **No speaker activity detection** - System doesn't know when it's speaking

---

## Solution Strategies (Prioritized)

### 🚀 Solution 1: Mute Microphone During TTS (EASIEST & MOST EFFECTIVE)

**Concept:** Disable audio capture while the robot is speaking.

**Implementation:**
```python
class AudioCapture:
    def __init__(self):
        self.is_muted = False
        self.mute_lock = threading.Lock()
    
    def mute(self):
        """Mute the microphone"""
        with self.mute_lock:
            self.is_muted = True
    
    def unmute(self):
        """Unmute the microphone"""
        with self.mute_lock:
            self.is_muted = False
    
    def record_utterance(self):
        # Skip recording if muted
        if self.is_muted:
            return None
        # ... existing code
```

**Integration:**
```python
# In main_tool_based.py
def run_once(self):
    # ... transcribe and process ...
    
    # MUTE before speaking
    self.audio_capture.mute()
    
    # Speak
    self.text_to_speech.speak(explanation, blocking=True)
    
    # Wait a bit for audio to finish echoing
    time.sleep(0.5)
    
    # UNMUTE after speaking
    self.audio_capture.unmute()
```

**Pros:**
- ✅ Simple to implement
- ✅ 100% effective (can't capture what you don't record)
- ✅ No false positives
- ✅ No additional dependencies

**Cons:**
- ⚠️ Can't interrupt robot while it's speaking
- ⚠️ Must wait for speech to complete

**Recommendation:** **START WITH THIS** - It's the simplest and most reliable solution.

---

### 🎯 Solution 2: Speaker Activity Flag (ENHANCED VERSION)

**Concept:** Track when TTS is active and block recording during that time, plus a grace period.

**Implementation:**
```python
class TTSManager:
    def __init__(self):
        self.is_speaking = False
        self.speaking_lock = threading.Lock()
    
    def speak(self, text, blocking=True):
        with self.speaking_lock:
            self.is_speaking = True
        
        try:
            # Generate and play TTS
            self._actually_speak(text)
        finally:
            # Grace period for echo to dissipate
            time.sleep(0.5)
            with self.speaking_lock:
                self.is_speaking = False
    
    def is_currently_speaking(self):
        with self.speaking_lock:
            return self.is_speaking

# In audio capture
def record_utterance(self):
    # Don't record if robot is speaking
    if self.tts_manager.is_currently_speaking():
        print("⏸️  Pausing capture - robot is speaking")
        return None
    
    # ... proceed with recording
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Thread-safe
- ✅ Adds grace period for echo

**Cons:**
- ⚠️ Requires passing TTS manager to audio capture
- ⚠️ Still blocks interruption

---

### 🔧 Solution 3: Acoustic Echo Cancellation (AEC) (ADVANCED)

**Concept:** Use signal processing to subtract the known output signal from the input.

**Libraries:**
- `speex` - Speex echo cancellation
- `webrtc-audio-processing` - Google's WebRTC AEC
- `sounddevice` with callback mode

**Implementation Example:**
```python
import speexdsp

class AECCapture:
    def __init__(self):
        self.aec = speexdsp.EchoCanceller(
            frame_size=160,
            filter_length=1024,
            sample_rate=16000
        )
    
    def process_audio(self, input_audio, output_audio):
        """
        Remove echo from input_audio using known output_audio
        """
        cleaned = self.aec.process(
            input_frame=input_audio,
            echo_frame=output_audio
        )
        return cleaned
```

**Pros:**
- ✅ Allows interruption during speech
- ✅ Professional solution
- ✅ Used in real products (Alexa, Google Home)

**Cons:**
- ⚠️ Complex to implement correctly
- ⚠️ Requires access to output audio signal
- ⚠️ CPU intensive
- ⚠️ May not be perfect (residual echo)
- ⚠️ Latency concerns

---

### 🎤 Solution 4: Improved VAD with Energy Gating (COMPLEMENTARY)

**Concept:** Ignore audio that sounds like TTS (different energy profile than human speech).

**Implementation:**
```python
class SmartVAD:
    def __init__(self):
        self.tts_playing = False
        self.energy_history = []
    
    def is_likely_human_speech(self, audio_chunk):
        energy = calculate_energy(audio_chunk)
        
        # TTS has very consistent energy
        # Human speech has more variation
        if self.tts_playing:
            return False
        
        # Check energy variance
        self.energy_history.append(energy)
        if len(self.energy_history) > 10:
            variance = np.var(self.energy_history[-10:])
            
            # Human speech: high variance
            # TTS/Robot: low variance
            if variance < THRESHOLD:
                return False  # Probably TTS echo
        
        return True
```

**Pros:**
- ✅ Allows interruption
- ✅ No hard muting
- ✅ Can adapt to different voices

**Cons:**
- ⚠️ Not 100% reliable
- ⚠️ Requires tuning
- ⚠️ May miss quiet human speech

---

### 🔌 Solution 5: Hardware/Physical Solutions

**Option A: Separate Audio Devices**
- Use Waveshare for output only
- Use Mac's built-in mic for input
- No physical proximity = no feedback

**Option B: Directional Microphone**
- Point mic away from speakers
- Reduces pickup of own voice

**Option C: Push-to-Talk**
- User holds button to activate mic
- Simple, 100% effective
- Bad UX for voice-first interface

**Option D: Headset/Earbuds**
- Audio goes to earbuds, not speakers
- Mic is far from output
- Not practical for robot embodiment

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (30 minutes)
**Implement microphone muting during TTS**

1. Add mute/unmute methods to `AudioCapture`
2. Mute before TTS, unmute after
3. Add 0.5s grace period after speech
4. Test with various commands

**Expected result:** 100% elimination of feedback loop

```python
# In src/audio/capture.py
class AudioCapture:
    def __init__(self, config):
        # ... existing init ...
        self.is_muted = False
        self._mute_lock = threading.Lock()
    
    def mute(self):
        """Temporarily disable audio capture"""
        with self._mute_lock:
            self.is_muted = True
            print("🔇 Microphone muted")
    
    def unmute(self):
        """Re-enable audio capture"""
        with self._mute_lock:
            self.is_muted = False
            print("🔊 Microphone unmuted")
    
    def record_utterance(self):
        """Record audio, but return None if muted"""
        # Check if muted
        with self._mute_lock:
            if self.is_muted:
                return None
        
        # ... existing recording code ...

# In src/main_tool_based.py
def run_once(self):
    # ... capture and process ...
    
    # MUTE before speaking
    self.audio_capture.mute()
    
    try:
        # Speak (blocking)
        self.text_to_speech.speak(explanation, blocking=True)
        
        # Grace period for acoustic echo to dissipate
        time.sleep(0.5)
    finally:
        # UNMUTE for next command
        self.audio_capture.unmute()
```

---

### Phase 2: Enhanced (Optional, 2-3 hours)
**Add speaker activity tracking**

1. Create `SpeakerActivityTracker` class
2. Track TTS start/stop with timestamps
3. Add configurable grace period
4. Log when capture is blocked

**Expected result:** More robust, with better logging

---

### Phase 3: Advanced (Optional, 1-2 days)
**Implement true AEC**

1. Research `webrtc-audio-processing` or `speex`
2. Capture TTS output signal
3. Apply echo cancellation
4. Allow interruption during speech

**Expected result:** Can interrupt robot mid-sentence

---

## Configuration Options

### Add to `audio_config.yaml`:
```yaml
audio:
  # ... existing config ...
  
  echo_prevention:
    method: "mute"  # Options: "mute", "aec", "vad", "none"
    grace_period_seconds: 0.5  # Wait after TTS before unmuting
    allow_interruption: false  # Future: enable with AEC
```

---

## Testing Strategy

### Test Cases:
1. **Basic command** - Say "move left", verify no echo
2. **Long response** - Command that generates long TTS, verify silence
3. **Rapid commands** - Multiple commands quickly, verify no cross-talk
4. **Ambient noise** - Test with background sounds, verify no false triggers
5. **Interruption** (Phase 3) - Try speaking while robot talks

### Success Criteria:
- ✅ Robot never processes its own speech as input
- ✅ User can issue command immediately after robot finishes speaking
- ✅ No false "No audio captured" messages during valid speech
- ✅ System feels responsive (not too much dead time)

---

## Edge Cases to Handle

1. **TTS fails mid-speech** - Unmute in finally block
2. **Non-blocking TTS** - Track actual playback completion
3. **Background speech** - Only mute for robot TTS, not other sounds
4. **Multiple TTS calls** - Queue mute/unmute properly
5. **User speaks during robot speech** - Phase 3 AEC solution

---

## Alternative: State Machine Approach

Add explicit states to prevent feedback:

```python
class SystemState(Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    GRACE_PERIOD = "grace_period"

# Only capture audio in LISTENING state
def should_capture_audio(self):
    return self.state == SystemState.LISTENING

def run_once(self):
    self.state = SystemState.LISTENING
    audio = self.audio_capture.record_utterance()
    
    self.state = SystemState.PROCESSING
    text = self.stt.transcribe(audio)
    tool_calls = self.llm.process(text)
    
    self.state = SystemState.SPEAKING
    self.tts.speak(explanation)
    
    self.state = SystemState.GRACE_PERIOD
    time.sleep(0.5)
    
    self.state = SystemState.LISTENING
```

---

## Comparison Matrix

| Solution | Complexity | Effectiveness | Allows Interruption | CPU Cost | Recommendation |
|----------|-----------|---------------|-------------------|----------|----------------|
| **Mute During TTS** | Low | 100% | No | None | ⭐ **Start here** |
| **Activity Flag** | Low-Med | 100% | No | None | ✅ Good enhancement |
| **AEC** | High | 90-95% | Yes | High | 🔮 Future upgrade |
| **Smart VAD** | Medium | 70-80% | Yes | Medium | 🤔 Complementary |
| **Separate Devices** | Low | 100% | Yes | None | 💰 Costs money |
| **Push-to-Talk** | Low | 100% | N/A | None | 😞 Bad UX |

---

## Recommendation

### Immediate (DO NOW):
✅ **Implement Solution 1: Microphone Muting**
- Simple, effective, reliable
- 30-minute implementation
- Zero additional dependencies
- Solves the problem completely

### Short-term Enhancement (NEXT WEEK):
✅ **Add Solution 2: Activity Tracking**
- Better architecture
- Easier to debug
- Foundation for future features

### Long-term (IF NEEDED):
🔮 **Consider Solution 3: AEC**
- Only if interruption is critical
- Research-heavy
- May not be worth complexity for single-user robot

---

## Implementation Checklist

- [ ] Add `mute()` and `unmute()` to `AudioCapture`
- [ ] Add thread-safe muting flag
- [ ] Modify `record_utterance()` to check mute status
- [ ] Update `main_tool_based.py` to mute before TTS
- [ ] Add grace period after TTS (0.5s)
- [ ] Use try/finally to ensure unmuting
- [ ] Test with various commands
- [ ] Add configuration for grace period
- [ ] Document behavior in README
- [ ] Add unit tests for mute/unmute

---

## Expected User Experience

### Before (Current):
```
User: "Move left"
Robot: "Moving left" 
Robot: [hears own voice]
Robot: "I'm not sure what you want me to do"
User: 😤
```

### After (With Muting):
```
User: "Move left"
Robot: [mutes mic] "Moving left" [unmutes mic]
[0.5s silence]
Robot: [ready for next command]
User: "Grasp"
Robot: [mutes mic] "Closing gripper" [unmutes mic]
User: 😊
```

---

## Next Steps

1. **Implement Phase 1 muting** (30 min)
2. **Test thoroughly** (15 min)
3. **Commit changes** with clear documentation
4. **Monitor for any issues** in real use
5. **Consider Phase 2 enhancements** if needed

**The muting solution should completely eliminate the feedback problem with minimal code changes.**

