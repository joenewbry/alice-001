#!/usr/bin/env python3
"""
Feedback system - coordinates robot gestures with speech output.
Provides synchronized visual (gesture) and auditory (speech) feedback.
"""

import threading
import time
from typing import Optional
from audio import TextToSpeech
from robot import HiwonderS1Controller


class FeedbackSystem:
    """Coordinates gesture and speech feedback"""
    
    def __init__(self, robot: HiwonderS1Controller, tts: TextToSpeech):
        """
        Initialize feedback system.
        
        Args:
            robot: Robot controller
            tts: Text-to-speech system
        """
        self.robot = robot
        self.tts = tts
        self.is_busy = False
        
        print("✓ Feedback system initialized")
    
    def nod_yes(self, speak: bool = False, text: str = "Yes"):
        """
        Perform 'yes' gesture (nod).
        
        Args:
            speak: Whether to speak confirmation
            text: Text to speak
        """
        self.is_busy = True
        
        try:
            if speak:
                # Speak and gesture simultaneously
                speech_thread = threading.Thread(target=self.tts.speak, args=(text, True))
                speech_thread.start()
                
                # Small delay so gesture follows speech
                time.sleep(0.2)
                self.robot.perform_gesture('nod')
                
                speech_thread.join()
            else:
                # Just gesture
                self.robot.perform_gesture('nod')
        
        finally:
            self.is_busy = False
    
    def shake_no(self, speak: bool = False, text: str = "No"):
        """
        Perform 'no' gesture (head shake).
        
        Args:
            speak: Whether to speak
            text: Text to speak
        """
        self.is_busy = True
        
        try:
            if speak:
                # Speak and gesture simultaneously
                speech_thread = threading.Thread(target=self.tts.speak, args=(text, True))
                speech_thread.start()
                
                time.sleep(0.2)
                self.robot.perform_gesture('shake')
                
                speech_thread.join()
            else:
                # Just gesture
                self.robot.perform_gesture('shake')
        
        finally:
            self.is_busy = False
    
    def acknowledge_valid_command(self, command_text: str):
        """
        Acknowledge a valid command with nod and speech.
        
        Args:
            command_text: The command response text
        """
        print(f"✓ Acknowledging valid command")
        
        # Nod yes
        self.robot.perform_gesture('nod')
        time.sleep(0.5)
        
        # Speak confirmation
        self.tts.speak(command_text)
    
    def acknowledge_invalid_command(self, error_message: str, suggestion: Optional[str] = None):
        """
        Acknowledge an invalid command with shake and helpful message.
        
        Args:
            error_message: What went wrong
            suggestion: Helpful suggestion from LLM
        """
        print(f"❌ Acknowledging invalid command")
        
        # Shake no
        self.robot.perform_gesture('shake')
        time.sleep(0.5)
        
        # Speak error and suggestion
        if suggestion:
            full_message = f"I don't know how to do that. {suggestion}"
        else:
            full_message = f"I don't understand. {error_message}"
        
        self.tts.speak(full_message)
    
    def speak_with_gesture(self, text: str, gesture: Optional[str] = None):
        """
        Speak text while performing a gesture.
        
        Args:
            text: Text to speak
            gesture: Optional gesture name to perform simultaneously
        """
        self.is_busy = True
        
        try:
            if gesture:
                # Start speech in thread
                speech_thread = threading.Thread(target=self.tts.speak, args=(text, True))
                speech_thread.start()
                
                # Perform gesture
                time.sleep(0.3)  # Brief delay
                self.robot.perform_gesture(gesture)
                
                speech_thread.join()
            else:
                # Just speak
                self.tts.speak(text)
        
        finally:
            self.is_busy = False
    
    def execute_command_with_feedback(self, command_dict: dict, response_text: str):
        """
        Execute a command with appropriate feedback.
        
        Args:
            command_dict: Parsed command dictionary
            response_text: Confirmation text to speak
        """
        self.is_busy = True
        
        try:
            # First acknowledge
            self.acknowledge_valid_command(response_text)
            
            # Wait a moment
            time.sleep(0.5)
            
            # Execute the actual command
            success = self._execute_robot_command(command_dict)
            
            if not success:
                self.tts.speak("Sorry, command execution failed")
            
            return success
        
        finally:
            self.is_busy = False
    
    def _execute_robot_command(self, command_dict: dict) -> bool:
        """
        Execute the robot command.
        
        Args:
            command_dict: Command parameters
            
        Returns:
            True if successful
        """
        cmd_type = command_dict.get('type')
        action = command_dict.get('action')
        
        try:
            # Gesture commands
            if cmd_type == 'gesture':
                gesture_name = command_dict.get('gesture')
                return self.robot.perform_gesture(gesture_name)
            
            # System commands
            elif cmd_type == 'system':
                if action == 'center':
                    return self.robot.motion.center_all()
                elif action == 'emergency_stop':
                    self.robot.emergency_stop()
                    return True
                elif action == 'repeat':
                    # Handled by command parser
                    return True
            
            # Gripper commands
            elif cmd_type == 'gripper':
                if action == 'close':
                    return self.robot.motion.grasp()
                elif action == 'open':
                    return self.robot.motion.release()
            
            # Motion commands
            elif cmd_type in ['directional', 'joint_specific']:
                joint = command_dict.get('joint')
                direction = command_dict.get('direction')
                amount = command_dict.get('amount', 45)
                
                if cmd_type == 'directional':
                    return self.robot.motion.move_directional(direction, amount)
                else:
                    return self.robot.motion.move_joint(joint, direction, amount)
            
            return False
            
        except Exception as e:
            print(f"❌ Command execution error: {e}")
            return False
    
    def wait_until_ready(self, timeout: float = 10.0):
        """
        Wait until feedback system is ready (not busy).
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        while self.is_busy and (time.time() - start_time) < timeout:
            time.sleep(0.1)


def demo_feedback():
    """Demo feedback system"""
    import yaml
    from pathlib import Path
    
    print("=== Feedback System Demo ===\n")
    
    # Load configs
    config_path = Path(__file__).parent.parent / 'config'
    
    try:
        with open(config_path / 'robot_config.yaml', 'r') as f:
            robot_config = yaml.safe_load(f)['robot']
    except:
        robot_config = {
            'servos': {
                'base': {'id': 1, 'min_angle': -90, 'max_angle': 90, 'center': 0},
                'wrist': {'id': 4, 'min_angle': -90, 'max_angle': 90, 'center': 0},
            },
            'motion': {'small': 15, 'medium': 45, 'large': 75, 'default_speed': 800}
        }
    
    # Import modules
    from robot import HiwonderS1Controller
    from audio import TextToSpeech
    
    # Create robot and TTS (will use simulation mode without hardware)
    robot = HiwonderS1Controller(robot_config)
    
    try:
        tts = TextToSpeech(voice="alloy")
        
        # Create feedback system
        feedback = FeedbackSystem(robot, tts)
        
        # Demo scenarios
        print("\n1. Valid command acknowledgment:")
        feedback.acknowledge_valid_command("Moving left")
        time.sleep(2)
        
        print("\n2. Invalid command acknowledgment:")
        feedback.acknowledge_invalid_command(
            "Command not recognized",
            "Try saying 'move left' or 'please dance'"
        )
        time.sleep(2)
        
        print("\n3. Gesture with speech:")
        feedback.speak_with_gesture("Watch this!", "wave")
        
        print("\n✓ Demo complete")
        
    except Exception as e:
        print(f"⚠️ Demo error (TTS requires API key): {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    demo_feedback()

