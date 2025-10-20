# Voice-Controlled Robot Arm System

A complete voice control system for the HiWonder robot arm with natural language processing, gesture feedback, and LLM-powered command assistance.

## Features

- 🎤 **Voice Input**: Capture audio via Waveshare USB audio device with Voice Activity Detection
- 🗣️ **Speech Recognition**: OpenAI Whisper for accurate speech-to-text
- 🔊 **Text-to-Speech**: OpenAI TTS for natural voice feedback
- 🤖 **Gesture Library**: Pre-programmed gestures (dance, wiggle, nod, wave)
- 🧠 **LLM Assistant**: GPT-4 powered command suggestions for unrecognized inputs
- ✅ **Visual Feedback**: Robot nods "yes" for valid commands, shakes "no" for invalid
- 🎯 **State Machine**: Robust state management for control flow
- 🔒 **Safety Features**: Joint limits, speed limits, emergency stop

## System Architecture

```
User Voice → Audio Capture → Speech-to-Text → Command Parser
                                                     ↓
                                              State Machine
                                                     ↓
                      ┌─────────────────────────────┴─────────────────────────┐
                      ↓                                                       ↓
              Valid Command?                                         Invalid Command?
                      ↓                                                       ↓
              Nod "Yes" + TTS                                   Shake "No" + LLM Suggestion
                      ↓
              Execute Command
         (Motion or Gesture)
```

## Installation

### 1. Clone Repository

```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPEN_AI_KEY=your_openai_api_key_here

# Audio Device (optional, auto-detected)
AUDIO_DEVICE_NAME=Waveshare USB Audio

# Robot Port (optional)
ROBOT_PORT=/dev/tty.usbmodemSN234567892
```

### 4. Configure Hardware

- Connect HiWonder robot arm via USB
- Connect Waveshare USB audio device
- Ensure microphone and speakers are working

## Usage

### Run Main Application

```bash
cd src
python main.py
```

The system will:
1. Initialize all components
2. Center the robot arm
3. Greet you with "Hello! I'm ready for voice commands"
4. Start listening for commands

### Voice Commands

#### Basic Motion
- **"move left"** / **"move right"** - Rotate base
- **"move up"** / **"move down"** - Raise/lower arm
- **"move forward"** / **"move back"** - Extend/retract arm

#### Motion Amounts
- **"move left a little"** - Small movement (15°)
- **"move left"** - Medium movement (45°, default)
- **"move left a lot"** - Large movement (75°)

#### Joint-Specific
- **"rotate base left"**
- **"move shoulder up"**
- **"move elbow forward"**
- **"move wrist down"**

#### Gripper
- **"grasp"** / **"grab"** - Close gripper
- **"release"** / **"open gripper"** - Open gripper

#### Gestures
- **"please dance"** - Perform dance sequence
- **"please wiggle"** - Wiggle all joints
- **"please nod"** - Nod up and down

#### System Commands
- **"center"** / **"home"** - Return to center position
- **"stop"** - Emergency stop
- **"repeat"** - Repeat last command

### Testing Individual Components

#### Test Audio Capture
```bash
cd src/audio
python capture.py
```

#### Test Speech-to-Text
```bash
cd src/audio
python speech_to_text.py recording.wav
```

#### Test Text-to-Speech
```bash
cd src/audio
python text_to_speech.py "Hello robot"
```

#### Test Command Parsing
```bash
cd src/control
python ontology.py
```

#### Test LLM Assistant
```bash
cd src/llm
python command_assistant.py
```

#### Test Robot Gestures
```bash
cd src/robot
python hiwonder_driver.py
```

## Configuration

### Audio Configuration (`config/audio_config.yaml`)

```yaml
audio:
  sample_rate: 16000
  channels: 1
  vad:
    enabled: true
    aggressiveness: 3
  recording:
    silence_threshold: 0.5
    max_recording_duration: 10
```

### Robot Configuration (`config/robot_config.yaml`)

```yaml
robot:
  servos:
    base:
      id: 1
      min_angle: -90
      max_angle: 90
    # ... other servos
  motion:
    small: 15
    medium: 45
    large: 75
```

### Command Ontology (`config/commands.yaml`)

Defines all valid commands and their patterns. Customize to add new commands.

## Project Structure

```
audio-control/
├── src/
│   ├── audio/               # Audio capture, STT, TTS
│   ├── control/             # State machine, command parser, ontology
│   ├── llm/                 # LLM assistant for suggestions
│   ├── robot/               # Robot driver, gestures, motion
│   ├── feedback_system.py   # Coordinate gestures + speech
│   └── main.py              # Main application
├── config/                  # Configuration files
├── requirements.txt
├── .env                     # Environment variables (create this)
└── README.md
```

## Troubleshooting

### Audio Device Not Found
- Check device name: `python -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"`
- Update `AUDIO_DEVICE_NAME` in `.env`

### Robot Not Connecting
- Check USB connection
- Verify port: `ls /dev/tty.*`
- Update `ROBOT_PORT` in `.env`
- System will fall back to simulation mode if hardware not available

### Whisper Model Loading Slow
- First run downloads model (~140MB for base model)
- Subsequent runs load from cache
- Use smaller model: Change `model_name="tiny"` in `main.py`

### TTS Not Working
- Verify `OPEN_AI_KEY` in `.env`
- Check OpenAI account has credits
- Test: `python src/audio/text_to_speech.py "test"`

### Commands Not Recognized
- Speak clearly and close to microphone
- Check command in ontology: `python src/control/ontology.py`
- LLM will suggest alternatives for unrecognized commands

## Safety

- **Joint Limits**: All movements respect configured min/max angles
- **Speed Limits**: Default 800ms per movement, configurable
- **Emergency Stop**: Say "stop" or press Ctrl+C
- **Timeout Protection**: Returns to center after 60s idle
- **Collision Detection**: Monitor servo load (if hardware supports)

## Development

### Adding New Commands

1. Add pattern to `config/commands.yaml`
2. Update parsing logic in `src/control/ontology.py`
3. Update LLM prompt in `src/llm/prompts.py`

### Adding New Gestures

1. Create gesture in `src/robot/gestures.py`
2. Add to `GestureLibrary._create_<name>()` method
3. Update command ontology

### Customizing Voice

Change TTS voice in `src/main.py`:
```python
self.text_to_speech = TextToSpeech(voice="nova")  # alloy, echo, fable, onyx, nova, shimmer
```

## License

MIT License - See LICENSE file

## Credits

- **OpenAI**: Whisper (speech recognition), GPT-4 (LLM assistance), TTS
- **HiWonder**: Robot arm hardware and xarm library
- **Waveshare**: USB audio device

## Support

For issues, questions, or contributions, please open an issue on the project repository.

---

**Enjoy your voice-controlled robot arm! 🤖🎤**

