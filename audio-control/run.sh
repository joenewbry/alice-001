#!/bin/bash
# Simple run script for voice control system

cd "$(dirname "$0")"
source venv/bin/activate
cd src
echo "Starting Voice-Controlled Robot Arm System..."
echo "Press Ctrl+C to stop"
echo ""
python main.py

