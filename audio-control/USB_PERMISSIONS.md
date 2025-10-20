# USB Robot Arm Permissions on macOS

## Current Status

✅ **Robot Detected!**
- Manufacturer: Hiwonder
- Product: xArm  
- Serial: 920A101095D182300023D4D4
- Vendor ID: 0x0483
- Product ID: 0x5750

❌ **Permission Issue**
- macOS is blocking direct USB access
- Error: "open failed" when trying to connect

## Solutions

### Option 1: Run in Simulation Mode (Current)

The system runs perfectly in simulation mode:
- All voice commands work
- State machine functions
- LLM suggestions active
- No actual robot movement (safe for testing)

**How to use:**
```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
./run.sh
```

You'll see: "🔧 Running in simulation mode"

### Option 2: Grant USB Permissions (For Real Robot Control)

**Method A: Run with sudo (temporary)**
```bash
cd /Users/Joseph.Newbry@alaskaair.com/dev/alice-001/audio-control
source venv/bin/activate
cd src
sudo python main.py
```

**Method B: Grant Terminal USB Access (permanent)**

1. Open **System Settings**
2. Go to **Privacy & Security** → **Bluetooth & USB**
3. Enable access for **Terminal** or **iTerm**
4. Restart terminal and run normally

**Method C: Grant Python USB Access**

1. Open **System Settings**  
2. Go to **Privacy & Security** → **Input Monitoring**
3. Add Python to allowed apps
4. Restart and run normally

### Option 3: Install IOKit USB Access (Advanced)

For development, you can create a codeless kext to allow USB access without sudo.

## Verification

Test USB connection with:
```bash
cd audio-control
source venv/bin/activate
python -c "import xarm; c = xarm.Controller('USB'); print('Connected!')"
```

If you see "Connected!" → USB permissions are working!
If you see "open failed" → Still need permissions

## What Works Without Permissions

Everything except actual robot movement:
- ✅ Voice capture (Waveshare USB Audio)
- ✅ Speech recognition (Whisper)
- ✅ Command parsing
- ✅ LLM suggestions
- ✅ Text-to-speech
- ✅ Gesture sequences (simulated)
- ❌ Actual servo movement

## Current Device Status

Your audio device is also detected:
- Manufacturer: Solid State System Co.,Ltd.
- Product: USB PnP Audio Device
- This works without special permissions!

---

**Recommendation:** Start with simulation mode to test the voice system, then grant USB permissions when ready for actual robot control.

