# Plan: Real-Time Streaming Audio with OpenAI Realtime API

## Problem Statement

**Current latency: ~9 seconds** 😱
- User speaks → wait for silence → upload audio → transcribe → LLM → TTS → download → play
- Sequential processing with multiple round trips
- Batch mode (not streaming)

**Target latency: < 1 second** ⚡
- User speaks → stream audio → get response in real-time → speak immediately
- Continuous bidirectional communication
- Server-side VAD and processing

---

## Current Architecture (Slow)

```
User speaks
    ↓ [Wait for silence: 1-2s]
Audio capture complete
    ↓ [Upload audio: 0.5s]
Faster-Whisper transcription: 0.3-0.5s
    ↓
LLM processing: 0.5-1s
    ↓ [API call: 0.5s]
TTS generation: 1-2s
    ↓ [Download: 0.5s]
Playback starts
─────────────────────────────
TOTAL: 5-9 seconds
```

**Bottlenecks:**
1. ⚠️ Waiting for complete utterance (VAD)
2. ⚠️ Sequential API calls (STT → LLM → TTS)
3. ⚠️ Upload/download overhead
4. ⚠️ No overlapping/pipelining

---

## Solution: OpenAI Realtime API

### What is it?
OpenAI's Realtime API provides **end-to-end voice conversation** with:
- ✅ Streaming audio input (WebSocket)
- ✅ Server-side VAD (no local waiting)
- ✅ Streaming transcription
- ✅ Function calling (for robot tools)
- ✅ Streaming audio output
- ✅ Single persistent connection

### New Architecture (Fast)

```
User speaks
    ↓ [Stream immediately: 0ms wait]
Audio chunks → WebSocket → Server VAD
    ↓ [Partial transcription: real-time]
LLM processes as speech comes in
    ↓ [Function calls: immediate]
TTS streams back
    ↓ [Play as received: 200-300ms]
Robot hears response
─────────────────────────────
TOTAL: 0.5-1.5 seconds ⚡
```

**Key improvements:**
1. ✅ No waiting for silence (stream as you speak)
2. ✅ Server-side VAD (faster detection)
3. ✅ Parallel processing (overlapping operations)
4. ✅ Streaming output (start speaking immediately)

---

## Implementation Options

### Option A: Full Realtime API Migration (RECOMMENDED)

**Use OpenAI's native Realtime API for everything**

#### Pros:
- ⚡ **Sub-second latency** (200-500ms typical)
- 🎤 Handles audio streaming automatically
- 🔧 Built-in function calling support
- 📡 Single WebSocket connection
- 🎯 Server-side VAD (no local processing)
- 🗣️ Natural conversation flow
- 💰 Potentially cheaper (fewer API calls)

#### Cons:
- 🔄 Requires rewrite of audio pipeline
- 📚 New API to learn
- 🌐 Requires persistent internet connection
- 🆕 Relatively new API (less mature)

#### Architecture:
```python
import asyncio
from openai import AsyncOpenAI
import pyaudio

class RealtimeVoiceController:
    async def connect(self):
        # Open WebSocket connection
        self.ws = await self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview",
            modalities=["audio", "text"],
            voice="alloy"
        )
    
    async def stream_audio_input(self):
        # Stream microphone to server
        while True:
            audio_chunk = await self.capture_audio()
            await self.ws.input_audio_buffer.append(audio_chunk)
    
    async def handle_responses(self):
        # Receive and play streaming audio
        async for event in self.ws:
            if event.type == "response.audio.delta":
                self.play_audio_chunk(event.delta)
            elif event.type == "response.function_call_arguments.done":
                await self.execute_robot_tool(event)
    
    async def configure_tools(self):
        # Register robot control tools
        await self.ws.session.update(
            tools=[
                {
                    "type": "function",
                    "name": "rotate_base",
                    "description": "Rotate the robot base",
                    "parameters": {...}
                },
                # ... other tools
            ]
        )
```

---

### Option B: Streaming Whisper + Streaming TTS (Hybrid)

**Keep faster-whisper, but add streaming**

#### Use:
- `faster-whisper` with chunked input
- OpenAI TTS with streaming output
- Custom streaming pipeline

#### Pros:
- 🔧 Incremental improvement
- 💾 Local STT (privacy)
- 🔄 Easier migration

#### Cons:
- ⚠️ Still requires VAD waiting
- ⚠️ Not as fast as Realtime API
- 🔧 More complex to implement correctly

---

### Option C: WebRTC + Local Models (Fully Local)

**Use local streaming models for everything**

#### Use:
- Streaming Whisper (whisper.cpp with streaming)
- Local LLM (Llama, etc.)
- Local TTS (Coqui, Piper)

#### Pros:
- 🔒 Full privacy
- 💰 No API costs
- 📶 Works offline

#### Cons:
- ⚠️ Complex setup
- 💻 High compute requirements
- 🎯 Lower quality than OpenAI
- ⏱️ May not be faster (local processing)

---

## Recommended Approach: Realtime API

### Implementation Plan

#### Phase 1: Proof of Concept (4-6 hours)
1. ✅ Create new `realtime_voice_control.py`
2. ✅ Implement WebSocket connection
3. ✅ Stream audio from microphone
4. ✅ Receive and play streaming responses
5. ✅ Test basic conversation

**Goal:** Prove we can get < 1s latency

#### Phase 2: Tool Integration (4-6 hours)
1. ✅ Define robot tools in Realtime API format
2. ✅ Handle function call events
3. ✅ Connect to existing robot controller
4. ✅ Test robot commands via voice

**Goal:** Full robot control via streaming

#### Phase 3: Production Ready (2-4 hours)
1. ✅ Error handling and reconnection
2. ✅ State management
3. ✅ Logging and debugging
4. ✅ Configuration options
5. ✅ Documentation

**Goal:** Reliable, production-ready system

---

## Realtime API Code Example

### Basic Setup

```python
#!/usr/bin/env python3
"""
Real-time voice control using OpenAI Realtime API
"""

import asyncio
import pyaudio
import json
from openai import AsyncOpenAI
from typing import Dict, Any

class RealtimeRobotController:
    """
    Streams audio to/from OpenAI Realtime API for ultra-low latency
    """
    
    def __init__(self, robot_controller, api_key: str):
        self.robot = robot_controller
        self.client = AsyncOpenAI(api_key=api_key)
        self.ws = None
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 24000  # Realtime API uses 24kHz
        self.chunk_size = 4096
        
        # State
        self.is_running = False
    
    async def connect(self):
        """Establish WebSocket connection"""
        print("🔌 Connecting to Realtime API...")
        
        self.ws = await self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01",
            modalities=["audio", "text"]
        )
        
        # Configure session
        await self.configure_session()
        
        print("✅ Connected!")
    
    async def configure_session(self):
        """Configure the session with voice, tools, and instructions"""
        
        # System instructions (your embodiment prompt)
        instructions = """You are controlling a 5-DOF robot arm...
        [Your full embodiment prompt here]
        """
        
        # Define robot tools
        tools = [
            {
                "type": "function",
                "name": "rotate_base",
                "description": "Rotate the robot base left or right",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {
                            "type": "number",
                            "description": "Degrees to rotate (negative=left, positive=right)"
                        }
                    },
                    "required": ["degrees"]
                }
            },
            {
                "type": "function",
                "name": "move_shoulder",
                "description": "Move the shoulder joint up or down",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {
                            "type": "number",
                            "description": "Degrees to move (negative=down, positive=up)"
                        }
                    },
                    "required": ["degrees"]
                }
            },
            # ... add all your robot tools
        ]
        
        await self.ws.session.update(
            session={
                "instructions": instructions,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500  # Much faster than local VAD!
                },
                "tools": tools,
                "temperature": 0.6
            }
        )
    
    async def stream_audio_input(self):
        """Stream microphone audio to server"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 Streaming microphone...")
        
        try:
            while self.is_running:
                # Read audio chunk
                audio_chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Stream to server
                await self.ws.input_audio_buffer.append(audio_chunk)
                
                await asyncio.sleep(0.01)  # Small delay
        finally:
            stream.stop_stream()
            stream.close()
    
    async def stream_audio_output(self):
        """Receive and play streaming audio from server"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🔊 Streaming audio output...")
        
        try:
            async for event in self.ws:
                # Handle different event types
                if event.type == "response.audio.delta":
                    # Stream audio to speakers immediately!
                    stream.write(event.delta)
                
                elif event.type == "response.function_call_arguments.done":
                    # Execute robot tool
                    await self.handle_function_call(event)
                
                elif event.type == "conversation.item.created":
                    print(f"📝 {event.item.role}: {event.item.content}")
                
                elif event.type == "response.done":
                    print("✅ Response complete")
                
                elif event.type == "error":
                    print(f"❌ Error: {event.error}")
        finally:
            stream.stop_stream()
            stream.close()
    
    async def handle_function_call(self, event):
        """Execute robot tool based on function call"""
        function_name = event.name
        arguments = json.loads(event.arguments)
        
        print(f"🤖 Executing: {function_name}({arguments})")
        
        # Map to your robot tools
        if function_name == "rotate_base":
            self.robot.move_servo(6, arguments['degrees'])
        elif function_name == "move_shoulder":
            self.robot.move_servo(5, arguments['degrees'])
        # ... etc
        
        # Send confirmation back to conversation
        await self.ws.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": event.call_id,
                "output": json.dumps({"status": "success"})
            }
        )
    
    async def run(self):
        """Main loop"""
        self.is_running = True
        
        await self.connect()
        
        # Run input and output streams concurrently
        await asyncio.gather(
            self.stream_audio_input(),
            self.stream_audio_output()
        )


async def main():
    from robot import HiwonderS1Controller
    
    # Initialize robot
    robot = HiwonderS1Controller(config)
    
    # Create realtime controller
    controller = RealtimeRobotController(
        robot_controller=robot,
        api_key=os.getenv("OPEN_AI_KEY")
    )
    
    # Run
    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Expected Performance

### Current System:
```
User: "Move left"
[9 seconds later]
Robot: "Moving left"
```

### Realtime API System:
```
User: "Move left"
[0.5 seconds - robot starts speaking]
Robot: "Moving left"
[Robot is already moving]
```

**18x faster!** 🚀

---

## Migration Strategy

### Option 1: Clean Break
- Create new `src/realtime_main.py`
- Keep old system for comparison
- Switch once validated

### Option 2: Gradual Migration
- Add realtime as optional mode
- Add `--realtime` flag to enable
- Maintain both systems

### Option 3: A/B Testing
- Run both systems
- Measure latency differences
- Gather user feedback

---

## Cost Comparison

### Current System (Per Command):
- Whisper: Local (free)
- GPT-4o-mini: ~$0.0001
- TTS: ~$0.0001
- **Total: ~$0.0002 per command**

### Realtime API:
- Real-time: ~$0.06/minute = ~$0.001 per command (10s)
- **Total: ~$0.001 per command**

**~5x more expensive, but worth it for 18x speedup!**

---

## Implementation Checklist

### Phase 1: Basic Streaming
- [ ] Install `openai` with Realtime support (latest version)
- [ ] Create `src/realtime_voice_control.py`
- [ ] Implement WebSocket connection
- [ ] Stream microphone audio
- [ ] Play streaming audio output
- [ ] Test basic conversation

### Phase 2: Robot Integration
- [ ] Define all robot tools in API format
- [ ] Handle function_call events
- [ ] Connect to HiwonderS1Controller
- [ ] Test all robot movements
- [ ] Add queue support for multiple commands

### Phase 3: Production
- [ ] Add error handling and reconnection
- [ ] Implement graceful shutdown
- [ ] Add logging and metrics
- [ ] Create configuration file
- [ ] Update documentation
- [ ] Add startup script

---

## Testing Plan

1. **Latency Test**: Measure time from speech start to response start
2. **Accuracy Test**: Verify all commands work correctly
3. **Reliability Test**: Run for extended periods
4. **Error Recovery**: Test connection failures
5. **Concurrent Commands**: Test queue handling

---

## Fallback Strategy

If Realtime API doesn't work well:

### Plan B: Parallel Processing
Keep current system but parallelize:
```python
async def process_command(audio):
    # Run transcription and TTS prep in parallel
    text_task = asyncio.create_task(stt.transcribe(audio))
    
    text = await text_task
    tool_calls_task = asyncio.create_task(llm.process(text))
    
    # Start TTS as soon as we know what to say
    tool_calls = await tool_calls_task
    
    # Fire robot and TTS in parallel
    await asyncio.gather(
        robot.execute(tool_calls),
        tts.speak(explanation)
    )
```

**Expected: 3-4s latency (better than 9s!)**

---

## Recommendations

### For immediate improvement (TODAY):
1. ✅ **Switch to Realtime API**
   - Biggest impact (18x faster)
   - Clean implementation
   - Best UX

### If Realtime API not available:
2. ✅ **Debug current system**
   - 9s is too slow even for old system
   - Should be ~1.5-2s with our optimizations
   - Check if faster-whisper is actually being used
   - Verify gpt-4o-mini (not gpt-4o)

### Next steps:
1. Create proof-of-concept with Realtime API
2. Measure actual latency
3. If < 1.5s, migrate fully
4. If not, investigate bottleneck in current system

---

## Resources

- [OpenAI Realtime API Docs](https://platform.openai.com/docs/guides/realtime)
- [Realtime API Reference](https://platform.openai.com/docs/api-reference/realtime)
- [Example Projects](https://github.com/openai/openai-realtime-api-beta)

---

## Next Step

**Create `src/realtime_main.py` and test!** 

Expected timeline:
- 2 hours: Basic proof-of-concept
- 4 hours: Full robot integration
- Result: < 1 second latency ⚡

Let's do this! 🚀

