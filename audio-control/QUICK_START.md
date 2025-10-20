# Quick Start Guide

## 🚀 Start the Voice Control System

```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
./run.sh
```

Or manually:
```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
source venv/bin/activate
cd src && python main.py
```

## 🎤 What You'll See

1. **System starts** (10 seconds)
   - Loads Whisper model
   - Initializes components
   - Connects to robot (or runs in simulation)

2. **Robot greets you**
   - Says "Hello! I'm ready for voice commands"
   
3. **Listening indicator**
   - "👂 Listening for voice command..."
   - Speak now!

## 🗣️ Voice Commands to Try

### Basic Motion
- "move left" - Rotate base left
- "move right a lot" - Large rotation
- "move up" - Raise arm
- "move down a little" - Lower arm slightly

### Gripper
- "grasp" - Close gripper
- "release" - Open gripper

### Fun Gestures
- "please dance" 🕺 - Dance sequence!
- "please wiggle" - Wiggle all joints
- "please nod" - Nod gesture

### System
- "center" - Return to home
- "stop" - Emergency stop
- "repeat" - Repeat last command

## 🤖 Robot Feedback

- **Nods "yes"** ✅ - Command understood
- **Shakes "no"** ❌ - Command not recognized
- **Speaks** - Confirms or suggests alternatives

## 🛑 Stop the System

Press **Ctrl+C** to stop

The robot will:
1. Say "Goodbye!"
2. Return to center position
3. Shut down gracefully

## 📝 Current Status

- ✅ Audio system ready
- ✅ Voice recognition (Whisper) ready
- ✅ LLM assistant ready
- ⚠️ Robot in simulation mode (USB permissions needed for real control)

See `USB_PERMISSIONS.md` for how to enable actual robot control.

## 🐛 Troubleshooting

**No audio captured?**
- Check microphone permissions
- Speak closer to microphone
- Check audio device is connected

**Commands not recognized?**
- Speak clearly
- Use exact command phrases
- Check examples above

**Want to see what's happening?**
- Watch terminal output
- All actions are logged with emojis

---

**Ready?** Run `./run.sh` and start talking to your robot! 🎤🤖
