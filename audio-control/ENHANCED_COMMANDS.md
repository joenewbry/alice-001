# 🎯 Enhanced Command System - Now Live!

## ✨ What's New

### 1. 📐 Precise Degree Control (NEW!)
You can now command exact degrees for ANY joint:

```
"rotate base 45 degrees left"
"move shoulder 60 degrees up"
"turn elbow 30 degrees forward"
"rotate wrist 15 degrees right"
```

**Works with any number!** The system will parse:
- Numbers: "45", "60", "30", "15"
- Words: "forty-five", "sixty", "thirty" (via LLM)
- Directions: left, right, up, down, forward, back

### 2. 🤖 LLM-Powered Fallback
If the regular parser doesn't understand, GPT-4 takes over!

**Examples it handles:**
- "rotate the base forty-five degrees to the left"
- "move the shoulder up by thirty degrees"
- "please turn the elbow 60 degrees forward"
- "can you rotate the wrist 25 degrees clockwise?"

The LLM converts natural language to structured commands automatically.

### 3. 📚 Complete Command Reference at Boot
Every time you start the system, you'll see a full reference guide showing:
- All available commands
- Example usage
- Tips and tricks
- Beginner to advanced examples

### 4. 🏠 Better Home Position
Fixed the default position to be compact and natural instead of fully extended:
- **Shoulder**: Raised up (-45°)
- **Elbow**: Bent inward (-60°)
- **Gripper**: Slightly open (30)

---

## 🎮 How to Use

### Simple Commands (Beginner)
```
"move left"           → Rotate base 45° left
"move right a lot"    → Rotate base 75° right
"grasp"               → Close gripper
"center"              → Return to home
```

### Precise Commands (Intermediate)
```
"rotate base 45 degrees left"         → Exactly 45° left
"move shoulder 30 degrees up"         → Exactly 30° up
"turn elbow 60 degrees forward"       → Exactly 60° forward
```

### Natural Language (Advanced)
```
"please rotate the base forty-five degrees counterclockwise"
"move the shoulder joint up by thirty degrees"
"can you turn the wrist 25 degrees to the right?"
```

All work thanks to LLM parsing!

---

## 🔧 Technical Details

### Command Processing Pipeline
1. **Direct Pattern Matching** - Fast regex parsing
2. **Degree-Specific Parser** - Extracts numbers + joints
3. **Standard Ontology** - Predefined command patterns
4. **LLM Fallback** - GPT-4 for complex natural language

### Joint Control
| Joint | ID | Range | Center Position |
|-------|-----|-------|-----------------|
| Base | 1 | -90° to +90° | 0° (forward) |
| Shoulder | 2 | -90° to +90° | -45° (raised) |
| Elbow | 3 | -90° to +90° | -60° (bent) |
| Wrist | 4 | -90° to +90° | 0° (level) |
| Gripper | 5 | 0 to 100 | 30 (open) |

### Motion Amounts
- **Small**: 15° ("a little", "slightly")
- **Medium**: 45° (default)
- **Large**: 75° ("a lot", "very")
- **Custom**: ANY degree you specify!

---

## 📋 Complete Command List

### Degree Commands (NEW!)
- `rotate [joint] [number] degrees [direction]`
- `move [joint] [number] degrees [direction]`
- `turn [joint] [number] degrees [direction]`

### Directional Commands
- `move left/right/up/down/forward/back [amount]`

### Joint-Specific
- `rotate base left/right [amount]`
- `move shoulder/elbow/wrist up/down [amount]`

### Gripper
- `grasp`, `grab`, `close gripper`
- `release`, `open gripper`

### Gestures
- `please dance` - Full dance sequence
- `please wiggle` - Wiggle all joints
- `please nod` - Nod gesture
- `wave` - Wave hello

### System
- `center`, `home`, `reset` - Return to home
- `stop`, `emergency stop` - Halt immediately
- `repeat`, `do that again` - Repeat last command

---

## 🎯 Examples by Skill Level

### Beginner
Start with these:
```
"move left"
"move right"
"grasp"
"please dance"
"center"
```

### Intermediate
Add precision:
```
"move left a lot"
"rotate base left"
"move shoulder up"
"release"
```

### Advanced
Exact control:
```
"rotate base 45 degrees left"
"move shoulder 60 degrees up"
"turn elbow 30 degrees forward"
"rotate wrist 15 degrees right"
```

### Expert (Natural Language)
Say it naturally:
```
"please rotate the base forty-five degrees counterclockwise"
"move the shoulder joint up by thirty degrees"
"can you turn the elbow sixty degrees forward?"
"rotate the wrist twenty-five degrees to the right"
```

---

## 🚀 Try It Now!

```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
./run.sh
```

At startup, you'll see the full command reference!

Then try:
1. **"rotate base 45 degrees left"** - Test precise control
2. **"move shoulder 30 degrees up"** - Move another joint
3. **"please dance"** - Have some fun!
4. **"center"** - Return home

---

## 💡 Pro Tips

1. **Precise Control**: Always specify degrees for exact positioning
2. **Natural Language**: Don't worry about exact phrasing - LLM will parse it
3. **Experiment**: Try different ways of saying the same thing
4. **Safety First**: Use "center" to return to safe position anytime
5. **Emergency**: Say "stop" for immediate halt

---

**All commands now support both relative movements AND absolute degree control!** 🎉

