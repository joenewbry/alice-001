# Brainstorm: Reducing Audio Processing Latency

**Problem Statement**: There's noticeable lag between:
1. Speaking a command
2. Receiving audio feedback
3. Robot movement execution

The audio response and robot movement are not synchronized, creating a disjointed user experience.

---

## Current System Pipeline

```
User speaks → Audio capture → STT (Whisper) → LLM processing → TTS (OpenAI) → Speaker output
                  ↓
              Robot movement (queued)
```

**Estimated Current Latencies:**
- Audio capture (VAD detection): 0.5-2s
- Speech-to-Text (Whisper): 1-3s
- LLM processing (GPT-4): 0.5-2s
- Text-to-Speech (OpenAI): 1-2s
- Network round-trips: 0.5-1s total
- **Total: 4-10 seconds**

---

## Identified Bottlenecks

### 1. **Speech-to-Text (Whisper) - BIGGEST BOTTLENECK**
- Running OpenAI Whisper API remotely
- Network latency + processing time
- Large audio files being uploaded

### 2. **LLM Processing (GPT-4)**
- Function calling requires model reasoning
- Network round-trip to OpenAI
- Using GPT-4 (more capable but slower)

### 3. **Text-to-Speech (OpenAI TTS)**
- Network round-trip
- Audio file download
- MP3 decoding before playback

### 4. **Audio Capture (VAD)**
- Waiting for silence before processing
- Energy-based detection adds delay
- Conservative thresholds = longer wait

### 5. **Sequential Processing**
- Everything runs in series
- No parallelization or streaming

---

## Proposed Solutions (Prioritized by Impact)

### 🚀 HIGH IMPACT - Quick Wins

#### 1. **Use Faster Whisper Model**
**Current**: `whisper-base` via OpenAI API  
**Solution**: Local Faster-Whisper or Whisper-tiny
```python
# Install: pip install faster-whisper
from faster_whisper import WhisperModel

# GPU if available, CPU fallback
model = WhisperModel("base", device="cpu", compute_type="int8")
# 2-3x faster than standard Whisper
```
**Expected savings**: 1-2 seconds  
**Tradeoff**: Slightly lower accuracy on tiny model  
**Recommendation**: Try `faster-whisper` with base model first

#### 2. **Switch to Faster LLM**
**Current**: GPT-4 (high latency)  
**Solutions**:
- **GPT-4o-mini**: Much faster, still capable for simple commands
- **GPT-3.5-turbo**: 2-3x faster than GPT-4
- **Streaming responses**: Start speaking while LLM is thinking

```python
# Use mini model for faster response
model="gpt-4o-mini"  # Instead of gpt-4o

# Or enable streaming
response = client.chat.completions.create(
    model="gpt-4o-mini",
    stream=True,  # Get tokens as they arrive
    ...
)
```
**Expected savings**: 0.5-1.5 seconds  
**Tradeoff**: Slightly less capable on complex commands

#### 3. **Stream TTS Audio**
**Current**: Wait for full MP3, then play  
**Solution**: Stream audio as it's generated
```python
# OpenAI supports streaming TTS
response = client.audio.speech.create(
    model="tts-1",  # Faster than tts-1-hd
    voice="alloy",
    input=text,
    response_format="opus",  # Lower latency than mp3
)

# Stream to speaker as chunks arrive
for chunk in response.iter_bytes(chunk_size=4096):
    audio_stream.write(chunk)
```
**Expected savings**: 0.5-1 second (start speaking sooner)  
**Tradeoff**: None, better UX

#### 4. **Reduce Audio Capture Latency**
**Current**: Waiting for silence with conservative thresholds  
**Solutions**:
- Reduce silence timeout (2s → 1s)
- More aggressive VAD settings
- Use push-to-talk for instant capture

```yaml
# audio_config.yaml
capture:
  silence_duration: 1.0  # Down from 2.0
  min_speech_duration: 0.3  # Down from 0.5

vad:
  aggressiveness: 3  # Max sensitivity (was 1)
```
**Expected savings**: 0.5-1 second  
**Tradeoff**: Might cut off end of speech

#### 5. **Parallel Processing**
**Current**: STT → LLM → TTS (sequential)  
**Solution**: Start TTS as soon as we have confirmation text
```python
# Don't wait for full LLM response to start speaking
async def process_command(audio):
    # These run in parallel
    text = await stt(audio)
    
    # Immediately speak acknowledgment while processing
    asyncio.create_task(tts.speak("Working on it"))
    
    # Process in background
    tool_calls = await llm.process(text)
    execute_tools(tool_calls)
```
**Expected savings**: 1-2 seconds perceived latency  
**Tradeoff**: Robot moves after you hear response

---

### 🎯 MEDIUM IMPACT - Architecture Changes

#### 6. **Local STT Model**
Use entirely local speech recognition:
- **Vosk**: Lightweight, 50-100ms latency
- **Coqui STT**: Fast, open-source
- **Whisper.cpp**: Optimized C++ version

```bash
# Example: Vosk (very fast)
pip install vosk
# Download small model (50MB)
# 10-20x faster than Whisper API
```
**Expected savings**: 2-3 seconds  
**Tradeoff**: Lower accuracy, larger dependency

#### 7. **Cache Common Commands**
Pre-process frequent commands:
```python
COMMAND_CACHE = {
    "move left": [{"name": "rotate_base", "args": {"degrees": -45}}],
    "move right": [{"name": "rotate_base", "args": {"degrees": 45}}],
    "grasp": [{"name": "close_gripper", "args": {"amount": 80}}],
    # ...
}

# Skip LLM for cached commands
if text.lower() in COMMAND_CACHE:
    tool_calls = COMMAND_CACHE[text.lower()]
    # Instant response!
```
**Expected savings**: 1-2 seconds (for common commands)  
**Tradeoff**: Only works for exact matches

#### 8. **Decouple Audio from Robot**
**Current**: Robot moves after audio completes  
**Solution**: Start robot immediately, audio confirms in parallel

```python
# Fire and forget robot commands
async def execute_command(text):
    tool_calls = await llm.process(text)
    
    # These happen simultaneously
    asyncio.create_task(execute_robot(tool_calls))  # Immediate
    asyncio.create_task(tts.speak("Moving left"))   # Parallel
```
**Expected savings**: 0 (but feels more responsive)  
**Tradeoff**: Audio might lag behind actual movement

#### 9. **Use WebRTC VAD Properly**
Re-enable WebRTC VAD with tuned settings:
```python
# More aggressive VAD = faster detection
vad = webrtcvad.Vad(3)  # Most aggressive

# Smaller frame sizes = faster response
frame_duration = 10  # ms (was 30ms)
```
**Expected savings**: 0.3-0.5 seconds  
**Tradeoff**: May trigger on background noise

---

### 💡 ADVANCED - Longer Term

#### 10. **Wake Word Detection**
Use always-on wake word (e.g., "Hey Robot"):
- Only process audio after wake word
- Can use very fast local model
- Immediate response after wake word

```bash
pip install pvporcupine  # Picovoice wake word
```

#### 11. **Edge TPU / Neural Accelerator**
Use hardware acceleration for Whisper:
- Coral Edge TPU
- Apple Neural Engine (M1/M2)
- NVIDIA Jetson (if available)

**Expected savings**: 2-3x STT speedup

#### 12. **WebSocket Streaming Pipeline**
Stream audio chunks as they're captured:
- Don't wait for full utterance
- Start transcribing immediately
- Partial results guide next steps

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Switch to `gpt-4o-mini` instead of `gpt-4o`
2. ✅ Use `tts-1` instead of `tts-1-hd`
3. ✅ Reduce silence timeout to 1 second
4. ✅ Stream TTS audio
5. ✅ Parallel audio/robot execution

**Expected improvement: 3-4 seconds → feels much snappier**

### Phase 2: Architecture (4-6 hours)
1. ✅ Implement `faster-whisper` locally
2. ✅ Add command caching for common phrases
3. ✅ Async/parallel processing pipeline
4. ✅ Immediate acknowledgment ("Got it") before processing

**Expected improvement: Sub-2 second perceived latency**

### Phase 3: Polish (optional)
1. ✅ Wake word detection
2. ✅ Hardware acceleration
3. ✅ Streaming STT
4. ✅ Predictive queuing

---

## Metrics to Track

```python
# Add timing instrumentation
import time

class LatencyTracker:
    def __init__(self):
        self.timings = {}
    
    def measure(self, stage):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                print(f"⏱️  {stage}: {elapsed:.2f}s")
                self.timings[stage] = elapsed
                return result
            return wrapper
        return decorator

# Usage
@tracker.measure("STT")
async def transcribe(audio):
    ...

# Track end-to-end latency
# Goal: < 2 seconds from speech end to audio start
```

---

## Example: Optimized Flow

```python
async def optimized_voice_loop():
    # 1. Audio capture (0.5s) - aggressive VAD
    audio = await capture.record_utterance(silence_timeout=1.0)
    
    # 2. STT (0.3s) - faster-whisper local
    text = await faster_whisper.transcribe(audio)
    
    # 3. Immediate acknowledgment (0.1s start)
    tts_task = asyncio.create_task(tts.speak_streamed("Got it"))
    
    # 4. Check cache first (0.001s)
    if text in command_cache:
        tool_calls = command_cache[text]
    else:
        # 5. LLM processing (0.5s) - gpt-4o-mini
        tool_calls = await llm.process(text)
    
    # 6. Execute robot (parallel with audio)
    robot_task = asyncio.create_task(queue.execute(tool_calls))
    
    # 7. Confirmation audio (parallel)
    await tts_task  # Already started
    
    # Total perceived latency: ~1 second
    # (audio starts playing at 0.9s, robot moves at 1.4s)
```

---

## Hardware Considerations

### Current Setup
- **Waveshare USB Audio**: Good quality, standard latency
- **Jetson (robot)**: Has GPU for potential acceleration
- **Mac (dev)**: M1/M2 has Neural Engine for Whisper

### Optimization Opportunities
1. **Run Whisper on Jetson GPU**: Offload from main machine
2. **Use Mac Neural Engine**: ~5x faster Whisper inference
3. **Dedicated audio buffer**: Reduce USB latency

---

## Conclusion

**Most impactful changes (implement first):**
1. ✅ Local `faster-whisper` (2-3s savings)
2. ✅ Switch to `gpt-4o-mini` (1s savings)
3. ✅ Stream TTS audio (0.5s perceived savings)
4. ✅ Parallel execution (1-2s perceived savings)
5. ✅ Reduce VAD timeout (0.5s savings)

**Target**: Reduce 6-8 second latency to **under 2 seconds**

**Next step**: Implement Phase 1 quick wins and measure impact.

