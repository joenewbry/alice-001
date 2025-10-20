#!/usr/bin/env python3
"""
Command parser - integrates ontology with state machine.
Validates and processes voice commands.
"""

from typing import Dict, Optional, Any
from .ontology import CommandOntology, Command, CommandType


class CommandParser:
    """Parses and validates voice commands"""
    
    def __init__(self, ontology: Optional[CommandOntology] = None, use_llm_fallback: bool = True):
        """
        Initialize command parser.
        
        Args:
            ontology: Command ontology instance
            use_llm_fallback: Whether to use LLM as fallback for complex commands
        """
        self.ontology = ontology or CommandOntology()
        self.last_valid_command: Optional[Command] = None
        self.use_llm_fallback = use_llm_fallback
        self.llm_parser = None
        
        if use_llm_fallback:
            try:
                from llm.command_parser_llm import LLMCommandParser
                self.llm_parser = LLMCommandParser()
            except Exception as e:
                print(f"⚠️ LLM fallback not available: {e}")
                self.use_llm_fallback = False
        
        print("✓ Command parser initialized")
    
    def parse(self, text: str) -> tuple[Optional[Command], bool, Optional[str]]:
        """
        Parse text command.
        
        Args:
            text: Transcribed text
            
        Returns:
            (command, is_valid, error_message)
        """
        if not text or not text.strip():
            return None, False, "Empty command"
        
        # Try standard parsing first
        command = self.ontology.parse_command(text)
        
        # If standard parsing fails, try LLM fallback
        if not command and self.use_llm_fallback and self.llm_parser:
            print("🤖 Using LLM fallback parser...")
            llm_result = self.llm_parser.parse(text)
            if llm_result:
                command = self._llm_dict_to_command(llm_result, text)
        
        if not command:
            return None, False, f"Command not recognized: '{text}'"
        
        # Handle repeat command
        if command.type == CommandType.SYSTEM and command.action == 'repeat':
            if self.last_valid_command:
                command = self.last_valid_command
                print(f"🔁 Repeating last command: {command.raw_text}")
            else:
                return None, False, "No previous command to repeat"
        
        # Validate command
        is_valid, error = self.ontology.validate_command(command)
        
        if not is_valid:
            return command, False, error
        
        # Save valid command
        if command.type != CommandType.SYSTEM or command.action != 'repeat':
            self.last_valid_command = command
        
        return command, True, None
    
    def command_to_dict(self, command: Command) -> Dict[str, Any]:
        """
        Convert Command to dictionary for execution.
        
        Args:
            command: Command object
            
        Returns:
            Dictionary with execution parameters
        """
        result = {
            'type': command.type.value,
            'action': command.action,
            'raw_text': command.raw_text
        }
        
        if command.joint:
            result['joint'] = command.joint
        
        if command.direction:
            result['direction'] = command.direction
        
        if command.amount:
            result['amount'] = command.amount
        
        if command.gesture:
            result['gesture'] = command.gesture
        
        return result
    
    def _llm_dict_to_command(self, llm_dict: Dict[str, Any], raw_text: str) -> Optional[Command]:
        """Convert LLM parsed dict to Command object"""
        cmd_type = llm_dict.get('type')
        
        if cmd_type == 'move_joint':
            return Command(
                type=CommandType.JOINT_SPECIFIC,
                action='move_joint',
                joint=llm_dict.get('joint'),
                direction=llm_dict.get('direction'),
                amount=llm_dict.get('degrees', 45),
                raw_text=raw_text
            )
        elif cmd_type == 'gripper':
            action = llm_dict.get('action')
            return Command(
                type=CommandType.GRIPPER,
                action=action,
                joint='gripper',
                raw_text=raw_text
            )
        elif cmd_type == 'gesture':
            return Command(
                type=CommandType.GESTURE,
                action='perform_gesture',
                gesture=llm_dict.get('gesture_name'),
                raw_text=raw_text
            )
        elif cmd_type == 'system':
            return Command(
                type=CommandType.SYSTEM,
                action=llm_dict.get('action'),
                raw_text=raw_text
            )
        
        return None
    
    def get_response_text(self, command: Command, is_valid: bool, error: Optional[str] = None) -> str:
        """
        Generate response text for TTS.
        
        Args:
            command: Parsed command
            is_valid: Whether command is valid
            error: Error message if invalid
            
        Returns:
            Text to speak
        """
        if not is_valid:
            return f"I don't understand. {error or 'Please try again.'}"
        
        # Generate confirmation text
        if command.type == CommandType.GESTURE:
            responses = {
                'dance': "Dancing!",
                'wiggle': "Wiggling!",
                'nod': "Nodding!"
            }
            return responses.get(command.gesture, "Performing gesture")
        
        elif command.type == CommandType.SYSTEM:
            responses = {
                'center': "Centering servos",
                'emergency_stop': "Stopping",
                'repeat': "Repeating last command"
            }
            return responses.get(command.action, "Executing")
        
        elif command.type == CommandType.GRIPPER:
            if command.action == 'close':
                return "Grasping"
            else:
                return "Releasing"
        
        elif command.type in [CommandType.DIRECTIONAL, CommandType.JOINT_SPECIFIC]:
            amount_text = {
                15: "a little",
                45: "",
                75: "a lot"
            }.get(command.amount, "")
            
            if command.type == CommandType.DIRECTIONAL:
                return f"Moving {command.direction} {amount_text}".strip()
            else:
                return f"Moving {command.joint} {command.direction} {amount_text}".strip()
        
        return "Executing command"


def demo_parser():
    """Demo command parser"""
    parser = CommandParser()
    
    test_commands = [
        "move left a little",
        "grasp",
        "please dance",
        "center",
        "move shoulder up a lot",
        "repeat",
        "make it fly"  # Invalid
    ]
    
    print("=== Testing Command Parser ===\n")
    
    for text in test_commands:
        print(f"Input: \"{text}\"")
        command, is_valid, error = parser.parse(text)
        
        if is_valid and command:
            print(f"  ✓ Valid command")
            cmd_dict = parser.command_to_dict(command)
            print(f"    Dict: {cmd_dict}")
            response = parser.get_response_text(command, is_valid)
            print(f"    Response: \"{response}\"")
        else:
            print(f"  ❌ Invalid: {error}")
            if command:
                response = parser.get_response_text(command, is_valid, error)
                print(f"    Response: \"{response}\"")
        
        print()


if __name__ == "__main__":
    demo_parser()

