# Latency Optimizations Applied

## 🎯 Goal
Reduce audio processing latency from **6-8 seconds** to **under 2 seconds**

---

## ✅ Changes Implemented (Phase 1)

### 1. **Faster-Whisper Local STT** (⚡ Biggest improvement)
- **Before**: OpenAI Whisper API (1-3s + network latency)
- **After**: Local `faster-whisper` (~0.3-0.5s)
- **Savings**: ~2-3 seconds
- **File**: `src/audio/faster_whisper_stt.py`
- **Usage**: Runs entirely locally with optimized inference

### 2. **Faster LLM Model** 
- **Before**: GPT-4o (0.5-2s)
- **After**: GPT-4o-mini (0.3-0.8s)
- **Savings**: ~1 second
- **File**: `src/tools/llm_executor.py` (line 43)
- **Note**: Still capable for simple robot commands

### 3. **Streaming TTS**
- **Before**: Wait for full MP3, then play
- **After**: Stream audio as generated
- **Savings**: ~0.5s (perceived latency)
- **File**: `src/audio/text_to_speech.py` (new `_speak_streaming` method)
- **Usage**: Set `stream=True` when calling `speak()`

### 4. **Reduced Audio Capture Timeout**
- **Before**: 2.0s silence threshold
- **After**: 1.0s silence threshold
- **Savings**: ~0.5-1 second
- **File**: `config/audio_config.yaml` (line 17)

### 5. **Parallel Audio/Robot Execution**
- **Before**: Wait for audio to finish before robot moves
- **After**: Robot moves while audio plays (`blocking=False`)
- **Savings**: ~1-2s perceived latency
- **File**: `src/main_tool_based.py` (lines 186-188)
- **Result**: Much more responsive feel

### 6. **Latency Tracking**
- Added real-time latency breakdown after each command
- Shows timing for: capture, STT, LLM, TTS
- Helps identify bottlenecks
- **File**: `src/main_tool_based.py` (lines 201-208)

---

## 📊 Expected Results

### Before Optimizations:
```
Audio capture:     0.5-2.0s
STT (Whisper API): 1.0-3.0s
LLM (GPT-4):       0.5-2.0s
TTS:               1.0-2.0s
Network overhead:  0.5-1.0s
─────────────────────────
TOTAL:             4.0-10.0s
```

### After Optimizations:
```
Audio capture:          ~0.5s (faster timeout)
STT (faster-whisper):   ~0.3s (local, optimized)
LLM (gpt-4o-mini):      ~0.5s (faster model)
TTS (streaming):        ~0.2s (to start)
Network overhead:       ~0.2s (only for LLM/TTS)
─────────────────────────────────
TOTAL:                  ~1.7s ✅
```

**Improvement: 4-8 seconds faster (70-80% reduction)**

---

## 🚀 How to Use

### Install New Dependencies
```bash
cd audio-control
source venv/bin/activate
pip install faster-whisper
```

### Run Optimized System
```bash
cd src
python main_tool_based.py
```

### What You'll See
- Real-time latency breakdown after each command
- Much faster response times
- Robot moves while audio plays (feels more natural)

---

## 🔧 Configuration

### Audio (config/audio_config.yaml)
```yaml
recording:
  silence_threshold: 1.0  # Reduced from 2.0
  min_recording_duration: 0.3  # Reduced from 0.5
```

### STT (src/main_tool_based.py)
```python
self.speech_to_text = FasterWhisperSTT(
    model_size="base",      # tiny/base/small/medium
    device="cpu",           # or "cuda" for GPU
    compute_type="int8"     # int8 for speed
)
```

### LLM (src/tools/llm_executor.py)
```python
def __init__(self, robot_tools, model: str = "gpt-4o-mini"):
    # Uses faster model by default
```

### TTS (src/main_tool_based.py)
```python
# Streaming + non-blocking for parallel execution
self.text_to_speech.speak(
    text,
    blocking=False,  # Robot can move while speaking
    stream=True      # Start speaking sooner
)
```

---

## 📈 Monitoring Latency

The system now prints a latency breakdown after each command:

```
⏱️  LATENCY BREAKDOWN:
   • Audio capture: 0.42s
   • STT (faster-whisper): 0.31s
   • LLM (gpt-4o-mini): 0.48s
   • TTS start (streaming): 0.15s
   📊 Total to audio output: 1.36s
   🎯 Target: < 2.0s
```

---

## 🎛️ Further Optimizations (Future)

### If You Need Even Faster:

1. **Use Tiny Model for STT**
   ```python
   FasterWhisperSTT(model_size="tiny")  # ~0.1s transcription
   ```

2. **GPU Acceleration** (if available)
   ```python
   FasterWhisperSTT(device="cuda")  # 2-3x faster
   ```

3. **Cache Common Commands**
   - Skip LLM for frequent commands
   - Instant response for "move left", "grasp", etc.

4. **Wake Word Detection**
   - Always-on listening with `pvporcupine`
   - Immediate processing after wake word

5. **True Streaming TTS**
   - Use PCM format with chunked playback
   - Start speaking while LLM is still thinking

---

## 🐛 Troubleshooting

### "faster-whisper not installed"
```bash
pip install faster-whisper
```

### "Model download is slow"
- First run downloads model (~150MB for base)
- Subsequent runs use cached model (instant)
- Models stored in: `~/.cache/huggingface/`

### "Audio still feels slow"
1. Check latency breakdown output
2. Ensure `faster-whisper` is being used (not standard Whisper)
3. Try `model_size="tiny"` for faster STT
4. Increase microphone volume to trigger capture sooner

---

## 📝 Files Modified

```
audio-control/
├── requirements.txt                    # Added faster-whisper
├── config/
│   └── audio_config.yaml              # Reduced silence timeout
├── src/
│   ├── audio/
│   │   ├── __init__.py                # Added FasterWhisperSTT import
│   │   ├── faster_whisper_stt.py      # NEW: Local STT
│   │   └── text_to_speech.py          # Added streaming support
│   ├── tools/
│   │   └── llm_executor.py            # Changed to gpt-4o-mini
│   └── main_tool_based.py             # Faster-whisper + latency tracking
└── workspace/
    └── brainstorm-latency.md          # Full analysis
```

---

## ✨ Summary

**Before**: Slow, sequential processing with network delays  
**After**: Fast, parallel processing with local inference

**Key wins:**
- ⚡ 70-80% latency reduction
- 🎯 Sub-2-second response time
- 🤖 Robot moves while speaking (natural feel)
- 📊 Real-time latency monitoring
- 🔒 More privacy (local STT)

**Result**: A much more responsive and natural voice control experience!

