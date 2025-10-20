#!/usr/bin/env python3
"""
LLM-powered command parser for handling complex natural language commands.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Any


class LLMCommandParser:
    """Use LLM to parse complex voice commands"""
    
    SYSTEM_PROMPT = """You are a command parser for a robot arm. Convert natural language to structured commands.

Available joints: base, shoulder, elbow, wrist, gripper

Available command types:
1. move_joint: Move a specific joint
   - Parameters: joint, direction (left/right/up/down/forward/back), degrees
2. gripper: Control gripper
   - Parameters: action (open/close)
3. gesture: Perform gesture
   - Parameters: gesture_name (dance/wiggle/nod)
4. system: System command
   - Parameters: action (center/stop/repeat)

Respond ONLY with valid JSON in this format:
{
  "type": "move_joint",
  "joint": "base",
  "direction": "left",
  "degrees": 45
}

Examples:
- "rotate the base 45 degrees to the left" → {"type": "move_joint", "joint": "base", "direction": "left", "degrees": 45}
- "move shoulder up by 30 degrees" → {"type": "move_joint", "joint": "shoulder", "direction": "up", "degrees": 30}
- "open the gripper" → {"type": "gripper", "action": "open"}
- "dance" → {"type": "gesture", "gesture_name": "dance"}

If command is unclear, respond with: {"type": "unknown"}
"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize LLM command parser"""
        load_dotenv()
        
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            raise ValueError("OPEN_AI_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        print(f"✓ LLM command parser initialized")
    
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse command using LLM.
        
        Args:
            text: Natural language command
            
        Returns:
            Parsed command dict or None
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Parse this command: {text}"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            parsed = json.loads(result)
            
            if parsed.get('type') == 'unknown':
                return None
            
            print(f"🤖 LLM parsed: {text} → {parsed}")
            return parsed
            
        except Exception as e:
            print(f"⚠️ LLM parsing failed: {e}")
            return None


def demo_llm_parser():
    """Demo LLM command parser"""
    parser = LLMCommandParser()
    
    test_commands = [
        "rotate the base 45 degrees to the left",
        "move the shoulder up by 30 degrees",
        "please rotate elbow 60 degrees forward",
        "open the gripper",
        "dance for me",
        "move all joints to zero position"
    ]
    
    print("=== Testing LLM Command Parser ===\n")
    
    for cmd in test_commands:
        print(f"Command: \"{cmd}\"")
        result = parser.parse(cmd)
        if result:
            print(f"  ✓ Parsed: {result}")
        else:
            print(f"  ❌ Could not parse")
        print()


if __name__ == "__main__":
    demo_llm_parser()

