#!/usr/bin/env python3
"""
Command reference - displays all available commands at startup.
"""

def get_command_reference() -> str:
    """Get comprehensive command reference text"""
    return """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      🤖 VOICE COMMAND REFERENCE                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📐 PRECISE DEGREE COMMANDS (NEW!)
────────────────────────────────────────────────────────────────────────────────
  "rotate base 45 degrees left"       - Rotate base 45° counterclockwise
  "rotate base 30 degrees right"      - Rotate base 30° clockwise
  "move shoulder 60 degrees up"       - Raise shoulder 60°
  "move elbow 20 degrees down"        - Lower elbow 20°
  "turn wrist 15 degrees left"        - Rotate wrist 15° left

  💡 Works with ANY joint and ANY degree amount!
  💡 Joints: base, shoulder, elbow, wrist, gripper

🎯 DIRECTIONAL COMMANDS
────────────────────────────────────────────────────────────────────────────────
  Basic:
    "move left" / "move right"        - Rotate base (default: 45°)
    "move up" / "move down"           - Raise/lower arm
    "move forward" / "move back"      - Extend/retract arm

  With Amount:
    "move left a little"              - Small movement (15°)
    "move right a lot"                - Large movement (75°)
    "move up moderately"              - Medium movement (45°)

🦾 JOINT-SPECIFIC COMMANDS
────────────────────────────────────────────────────────────────────────────────
  "rotate base left"                  - Rotate base
  "move shoulder up"                  - Move shoulder joint
  "move elbow forward"                - Move elbow joint
  "move wrist down"                   - Move wrist joint

  💡 Add amounts: "rotate base left a lot"

✋ GRIPPER COMMANDS
────────────────────────────────────────────────────────────────────────────────
  "grasp" / "grab"                    - Close gripper
  "release" / "open gripper"          - Open gripper
  "close gripper"                     - Close gripper fully

🎭 GESTURE COMMANDS
────────────────────────────────────────────────────────────────────────────────
  "please dance"                      - Full dance sequence (fun!)
  "please wiggle"                     - Wiggle all joints
  "please nod"                        - Nod up and down
  "wave"                              - Wave hello

⚙️  SYSTEM COMMANDS
────────────────────────────────────────────────────────────────────────────────
  "center" / "home" / "reset"         - Return to home position
  "stop" / "emergency stop"           - Halt all motion immediately
  "repeat" / "do that again"          - Repeat last command

🤖 ROBOT FEEDBACK
────────────────────────────────────────────────────────────────────────────────
  ✅ Nods "yes"     - When it understands your command
  ❌ Shakes "no"    - When command is not recognized
  🗣️  Speaks        - Confirms action or suggests alternatives

💬 NATURAL LANGUAGE (LLM-Powered)
────────────────────────────────────────────────────────────────────────────────
  The robot uses AI to understand complex commands like:
  • "rotate the base forty-five degrees to the left"
  • "move the shoulder up by thirty degrees"
  • "please rotate elbow 60 degrees forward"
  • "turn the wrist 25 degrees clockwise"

  💡 If unsure, just say it naturally - the LLM will help parse it!

📊 EXAMPLES
────────────────────────────────────────────────────────────────────────────────
  Beginner:
    "move left"                       - Simple directional
    "grasp"                           - Close gripper
    "center"                          - Return home

  Intermediate:
    "move right a lot"                - Large rotation
    "move shoulder up"                - Joint-specific
    "please dance"                    - Fun gesture

  Advanced:
    "rotate base 45 degrees left"     - Precise degree control
    "move elbow 60 degrees forward"   - Exact positioning
    "turn wrist 30 degrees right"     - Fine control

🎓 TIPS
────────────────────────────────────────────────────────────────────────────────
  • Speak clearly and naturally
  • Specify degrees for precise control
  • Use "a little" or "a lot" for relative movements
  • Say "repeat" to do the last command again
  • Say "center" if you get confused - returns to safe position
  • Say "stop" anytime for emergency halt

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Ready to begin! Start with simple commands and work up to precise control! ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

def print_command_reference():
    """Print the command reference"""
    print(get_command_reference())


if __name__ == "__main__":
    print_command_reference()

