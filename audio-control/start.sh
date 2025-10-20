#!/bin/bash
# Quick start script for voice-controlled robot arm

echo "======================================"
echo "Voice-Controlled Robot Arm System"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from example..."
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
# OpenAI API Key for Whisper and TTS
OPEN_AI_KEY=your_openai_api_key_here

# Audio Device Configuration
AUDIO_DEVICE_NAME=Waveshare USB Audio

# Robot Configuration
ROBOT_PORT=/dev/tty.usbmodemSN234567892
ROBOT_SIMULATION_MODE=false
EOF
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPEN_AI_KEY"
    echo ""
    read -p "Press Enter when ready to continue..."
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import pyaudio, whisper, openai, transitions, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing"
    echo "Installing from requirements.txt..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        echo "Try: pip3 install -r requirements.txt"
        exit 1
    fi
    echo "✓ Dependencies installed"
fi

echo ""
echo "🚀 Starting voice control system..."
echo ""

# Run the main application
cd src
python3 main.py

echo ""
echo "✓ System shut down"

