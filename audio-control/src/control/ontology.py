#!/usr/bin/env python3
"""
Command ontology - defines all valid robot commands and their parameters.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Types of commands"""
    DIRECTIONAL = "directional"
    JOINT_SPECIFIC = "joint_specific"
    GRIPPER = "gripper"
    GESTURE = "gesture"
    SYSTEM = "system"


class MotionAmount(Enum):
    """Motion amount levels"""
    SMALL = 15
    MEDIUM = 45
    LARGE = 75


@dataclass
class Command:
    """Parsed command structure"""
    type: CommandType
    action: str
    joint: Optional[str] = None
    direction: Optional[str] = None
    amount: int = 45  # Default to medium
    gesture: Optional[str] = None
    raw_text: str = ""
    confidence: float = 1.0


class CommandOntology:
    """
    Command ontology system that defines all valid commands
    and provides parsing functionality.
    """
    
    # Amount keywords mapping
    AMOUNT_KEYWORDS = {
        'small': ['small', 'little', 'bit', 'slightly', 'tiny', 'a little'],
        'medium': ['medium', 'moderate', 'normal', 'moderately'],
        'large': ['large', 'lot', 'big', 'much', 'very', 'a lot']
    }
    
    # Direction mappings
    DIRECTIONS = {
        'left': ['left', 'counterclockwise'],
        'right': ['right', 'clockwise'],
        'up': ['up', 'raise', 'lift'],
        'down': ['down', 'lower'],
        'forward': ['forward', 'extend', 'out'],
        'back': ['back', 'backward', 'retract', 'in']
    }
    
    # Joint names
    JOINTS = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ontology.
        
        Args:
            config: Optional command configuration from commands.yaml
        """
        self.config = config or {}
        print("✓ Command ontology initialized")
    
    def parse_amount(self, text: str) -> int:
        """
        Parse motion amount from text.
        
        Args:
            text: Input text
            
        Returns:
            Motion amount in degrees
        """
        text_lower = text.lower()
        
        # Check for amount keywords
        for amount_type, keywords in self.AMOUNT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return MotionAmount[amount_type.upper()].value
        
        # Default to medium
        return MotionAmount.MEDIUM.value
    
    def parse_direction(self, text: str) -> Optional[str]:
        """
        Parse direction from text.
        
        Args:
            text: Input text
            
        Returns:
            Direction string or None
        """
        text_lower = text.lower()
        
        for direction, keywords in self.DIRECTIONS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return direction
        
        return None
    
    def parse_joint(self, text: str) -> Optional[str]:
        """
        Parse joint name from text.
        
        Args:
            text: Input text
            
        Returns:
            Joint name or None
        """
        text_lower = text.lower()
        
        for joint in self.JOINTS:
            if joint in text_lower:
                return joint
        
        return None
    
    def parse_command(self, text: str) -> Optional[Command]:
        """
        Parse text into a Command object.
        
        Args:
            text: Transcribed text command
            
        Returns:
            Command object or None if not recognized
        """
        text = text.lower().strip()
        
        # Try to match each command type
        
        # 1. Degree-specific commands (new - highest priority)
        degree_cmd = self._parse_degree_command(text)
        if degree_cmd:
            return degree_cmd
        
        # 2. Gesture commands
        gesture = self._parse_gesture(text)
        if gesture:
            return gesture
        
        # 3. System commands
        system_cmd = self._parse_system_command(text)
        if system_cmd:
            return system_cmd
        
        # 4. Gripper commands
        gripper = self._parse_gripper(text)
        if gripper:
            return gripper
        
        # 5. Joint-specific commands
        joint_cmd = self._parse_joint_specific(text)
        if joint_cmd:
            return joint_cmd
        
        # 6. Directional commands
        directional = self._parse_directional(text)
        if directional:
            return directional
        
        # Command not recognized
        return None
    
    def _parse_degree_command(self, text: str) -> Optional[Command]:
        """Parse commands with specific degrees"""
        # Pattern: "rotate [joint] [degrees] degrees" or "move [joint] to [degrees] degrees"
        
        # Extract degrees from text
        degree_match = re.search(r'(\d+)\s*degrees?', text)
        if not degree_match:
            return None
        
        degrees = int(degree_match.group(1))
        
        # Find joint name
        joint = None
        for joint_name in self.JOINTS:
            if joint_name in text:
                joint = joint_name
                break
        
        if not joint:
            return None
        
        # Determine if it's absolute or relative
        is_relative = any(word in text for word in ['rotate', 'turn', 'move'])
        is_absolute = any(word in text for word in ['to', 'at', 'position'])
        
        # Determine direction for relative movements
        direction = None
        if is_relative:
            if any(word in text for word in ['left', 'counterclockwise', 'ccw']):
                direction = 'left'
                degrees = -degrees  # Negative for left
            elif any(word in text for word in ['right', 'clockwise', 'cw']):
                direction = 'right'
            elif any(word in text for word in ['up', 'raise', 'lift']):
                direction = 'up'
            elif any(word in text for word in ['down', 'lower']):
                direction = 'down'
                degrees = -degrees  # Negative for down
            elif any(word in text for word in ['forward', 'extend']):
                direction = 'forward'
            elif any(word in text for word in ['back', 'backward', 'retract']):
                direction = 'back'
                degrees = -degrees  # Negative for back
        
        return Command(
            type=CommandType.JOINT_SPECIFIC,
            action='move_joint',
            joint=joint,
            direction=direction,
            amount=abs(degrees),
            raw_text=text
        )
    
    def _parse_gesture(self, text: str) -> Optional[Command]:
        """Parse gesture commands"""
        gesture_patterns = {
            'dance': r'(please\s+)?(dance|dancing)',
            'wiggle': r'(please\s+)?(wiggle|wiggling)',
            'nod': r'(please\s+)?(nod|nodding)'
        }
        
        for gesture_name, pattern in gesture_patterns.items():
            if re.search(pattern, text):
                return Command(
                    type=CommandType.GESTURE,
                    action='perform_gesture',
                    gesture=gesture_name,
                    raw_text=text
                )
        
        return None
    
    def _parse_system_command(self, text: str) -> Optional[Command]:
        """Parse system commands"""
        if re.search(r'(center|home|reset)', text):
            return Command(
                type=CommandType.SYSTEM,
                action='center',
                raw_text=text
            )
        
        if re.search(r'(stop|halt|emergency)', text):
            return Command(
                type=CommandType.SYSTEM,
                action='emergency_stop',
                raw_text=text
            )
        
        if re.search(r'(repeat|again|do that again)', text):
            return Command(
                type=CommandType.SYSTEM,
                action='repeat',
                raw_text=text
            )
        
        return None
    
    def _parse_gripper(self, text: str) -> Optional[Command]:
        """Parse gripper commands"""
        if re.search(r'(grasp|grab|close\s+gripper|squeeze)', text):
            return Command(
                type=CommandType.GRIPPER,
                action='close',
                joint='gripper',
                raw_text=text
            )
        
        if re.search(r'(release|open\s+gripper|let\s+go|drop)', text):
            return Command(
                type=CommandType.GRIPPER,
                action='open',
                joint='gripper',
                raw_text=text
            )
        
        return None
    
    def _parse_joint_specific(self, text: str) -> Optional[Command]:
        """Parse joint-specific commands"""
        # Pattern: "move/rotate <joint> <direction> [amount]"
        joint = self.parse_joint(text)
        if not joint:
            return None
        
        # Check if it's a joint-specific command
        if not re.search(r'(move|rotate)\s+(base|shoulder|elbow|wrist)', text):
            return None
        
        direction = self.parse_direction(text)
        amount = self.parse_amount(text)
        
        if direction:
            return Command(
                type=CommandType.JOINT_SPECIFIC,
                action='move_joint',
                joint=joint,
                direction=direction,
                amount=amount,
                raw_text=text
            )
        
        return None
    
    def _parse_directional(self, text: str) -> Optional[Command]:
        """Parse directional movement commands"""
        # Pattern: "move <direction> [amount]"
        if not re.search(r'move\s+(left|right|up|down|forward|back)', text):
            return None
        
        direction = self.parse_direction(text)
        if not direction:
            return None
        
        amount = self.parse_amount(text)
        
        # Map direction to joint
        joint_map = {
            'left': 'base',
            'right': 'base',
            'up': 'shoulder',
            'down': 'shoulder',
            'forward': 'elbow',
            'back': 'elbow'
        }
        
        joint = joint_map.get(direction)
        
        return Command(
            type=CommandType.DIRECTIONAL,
            action='move',
            joint=joint,
            direction=direction,
            amount=amount,
            raw_text=text
        )
    
    def get_command_examples(self) -> List[str]:
        """Get list of example commands"""
        return [
            "move left",
            "move right a lot",
            "move up a little",
            "move down",
            "move forward",
            "move back",
            "rotate base left",
            "move shoulder up",
            "move elbow forward",
            "move wrist down",
            "grasp",
            "release",
            "please dance",
            "please wiggle",
            "please nod",
            "center",
            "stop",
            "repeat"
        ]
    
    def validate_command(self, command: Command, robot_state: Dict = None) -> tuple[bool, Optional[str]]:
        """
        Validate if command can be executed safely.
        
        Args:
            command: Command to validate
            robot_state: Current robot state (joint positions, etc.)
            
        Returns:
            (is_valid, error_message)
        """
        # Basic validation - could be extended with robot state checking
        
        if command.type == CommandType.SYSTEM:
            return True, None
        
        if command.type == CommandType.GESTURE:
            if command.gesture not in ['dance', 'wiggle', 'nod']:
                return False, f"Unknown gesture: {command.gesture}"
            return True, None
        
        if command.type in [CommandType.DIRECTIONAL, CommandType.JOINT_SPECIFIC]:
            if command.joint not in self.JOINTS:
                return False, f"Invalid joint: {command.joint}"
            
            if command.amount < 1 or command.amount > 90:
                return False, f"Motion amount out of range: {command.amount}"
            
            return True, None
        
        if command.type == CommandType.GRIPPER:
            return True, None
        
        return False, "Unknown command type"


def demo_ontology():
    """Demo command parsing"""
    ontology = CommandOntology()
    
    # Test commands
    test_commands = [
        "move left a little",
        "move right a lot",
        "please dance",
        "grasp",
        "release",
        "rotate base left",
        "move shoulder up",
        "center",
        "stop",
        "please wiggle",
        "move forward",
        "make it fly"  # Invalid
    ]
    
    print("=== Testing Command Parsing ===\n")
    
    for text in test_commands:
        print(f"Input: \"{text}\"")
        command = ontology.parse_command(text)
        
        if command:
            print(f"  ✓ Type: {command.type.value}")
            print(f"    Action: {command.action}")
            if command.joint:
                print(f"    Joint: {command.joint}")
            if command.direction:
                print(f"    Direction: {command.direction}")
            if command.type in [CommandType.DIRECTIONAL, CommandType.JOINT_SPECIFIC]:
                print(f"    Amount: {command.amount}°")
            if command.gesture:
                print(f"    Gesture: {command.gesture}")
            
            # Validate
            is_valid, error = ontology.validate_command(command)
            if is_valid:
                print(f"    ✓ Valid command")
            else:
                print(f"    ❌ Invalid: {error}")
        else:
            print(f"  ❌ Command not recognized")
        
        print()


if __name__ == "__main__":
    demo_ontology()

