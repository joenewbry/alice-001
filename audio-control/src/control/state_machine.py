#!/usr/bin/env python3
"""
State machine for voice-controlled robot arm.
Manages states: Idle, Listening, Processing, Validating, Understanding, Requesting, Speaking, Executing.
"""

from transitions import Machine
from typing import Optional, Callable, Dict, Any
import time
from dataclasses import dataclass
from enum import Enum


class RobotState(Enum):
    """Robot states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    VALIDATING_COMMAND = "validating_command"
    UNDERSTANDING_GESTURE = "understanding_gesture"
    REQUESTING_SUGGESTION = "requesting_suggestion"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    EXECUTION_COMPLETE = "execution_complete"


@dataclass
class CommandContext:
    """Context for command execution"""
    raw_audio: Optional[bytes] = None
    transcribed_text: Optional[str] = None
    parsed_command: Optional[Dict[str, Any]] = None
    is_valid: bool = False
    error_message: Optional[str] = None
    llm_suggestion: Optional[str] = None
    last_command: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0


class RobotStateMachine:
    """State machine for robot control flow"""
    
    # Define states
    states = [
        'idle',
        'listening',
        'processing',
        'validating_command',
        'understanding_gesture',
        'requesting_suggestion',
        'speaking',
        'executing',
        'execution_complete'
    ]
    
    # Define transitions
    transitions = [
        # From idle
        {'trigger': 'start_listening', 'source': 'idle', 'dest': 'listening'},
        
        # From listening
        {'trigger': 'audio_captured', 'source': 'listening', 'dest': 'processing'},
        {'trigger': 'timeout', 'source': 'listening', 'dest': 'idle'},
        
        # From processing
        {'trigger': 'transcription_complete', 'source': 'processing', 'dest': 'validating_command'},
        {'trigger': 'transcription_failed', 'source': 'processing', 'dest': 'listening'},
        
        # From validating
        {'trigger': 'command_recognized', 'source': 'validating_command', 'dest': 'understanding_gesture'},
        {'trigger': 'command_not_recognized', 'source': 'validating_command', 'dest': 'requesting_suggestion'},
        
        # From understanding gesture
        {'trigger': 'nod_complete', 'source': 'understanding_gesture', 'dest': 'speaking'},
        
        # From requesting suggestion
        {'trigger': 'suggestion_ready', 'source': 'requesting_suggestion', 'dest': 'speaking'},
        
        # From speaking (valid command)
        {'trigger': 'feedback_complete_valid', 'source': 'speaking', 'dest': 'executing'},
        {'trigger': 'feedback_complete_invalid', 'source': 'speaking', 'dest': 'listening'},
        
        # From executing
        {'trigger': 'motion_complete', 'source': 'executing', 'dest': 'execution_complete'},
        {'trigger': 'motion_failed', 'source': 'executing', 'dest': 'idle'},
        
        # From execution complete
        {'trigger': 'ready_for_next', 'source': 'execution_complete', 'dest': 'listening'},
        
        # Emergency transitions from any state
        {'trigger': 'emergency_stop', 'source': '*', 'dest': 'idle'},
    ]
    
    def __init__(self, callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize state machine.
        
        Args:
            callbacks: Dictionary of callback functions for various events
                - on_listening: Called when entering listening state
                - on_processing: Called when processing audio
                - on_command_valid: Called when command is valid
                - on_command_invalid: Called when command is invalid
                - on_speaking: Called when robot speaks
                - on_executing: Called when executing command
        """
        self.callbacks = callbacks or {}
        self.context = CommandContext()
        
        # Initialize state machine
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial='idle',
            auto_transitions=False,
            after_state_change='_on_state_change'
        )
        
        # Timeout tracking
        self.listening_timeout = 10.0  # seconds
        self.listening_start_time = 0.0
        
        print("✓ State machine initialized")
    
    def _on_state_change(self):
        """Called after every state transition"""
        print(f"🔄 State: {self.state}")
        
        # State-specific actions
        if self.state == 'listening':
            self.listening_start_time = time.time()
            self._callback('on_listening')
            
        elif self.state == 'processing':
            self._callback('on_processing', self.context)
            
        elif self.state == 'understanding_gesture':
            self._callback('on_command_valid', self.context)
            
        elif self.state == 'requesting_suggestion':
            self._callback('on_command_invalid', self.context)
            
        elif self.state == 'speaking':
            self._callback('on_speaking', self.context)
            
        elif self.state == 'executing':
            self._callback('on_executing', self.context)
    
    def _callback(self, name: str, *args, **kwargs):
        """Execute callback if registered"""
        if name in self.callbacks:
            try:
                self.callbacks[name](*args, **kwargs)
            except Exception as e:
                print(f"❌ Callback error '{name}': {e}")
    
    def check_listening_timeout(self) -> bool:
        """Check if listening state has timed out"""
        if self.state == 'listening':
            elapsed = time.time() - self.listening_start_time
            if elapsed > self.listening_timeout:
                print("⏱️ Listening timeout")
                self.timeout()
                return True
        return False
    
    def set_audio(self, audio_data: bytes):
        """Set captured audio data"""
        self.context.raw_audio = audio_data
        self.context.timestamp = time.time()
    
    def set_transcription(self, text: str):
        """Set transcribed text"""
        self.context.transcribed_text = text
    
    def set_command(self, command: Dict[str, Any], is_valid: bool):
        """Set parsed command"""
        self.context.parsed_command = command
        self.context.is_valid = is_valid
        
        if is_valid:
            # Save as last command for "repeat"
            self.context.last_command = command.copy()
    
    def set_error(self, error: str):
        """Set error message"""
        self.context.error_message = error
    
    def set_suggestion(self, suggestion: str):
        """Set LLM suggestion"""
        self.context.llm_suggestion = suggestion
    
    def get_context(self) -> CommandContext:
        """Get current context"""
        return self.context
    
    def reset_context(self):
        """Reset context for next command (preserve last_command)"""
        last_cmd = self.context.last_command
        self.context = CommandContext()
        self.context.last_command = last_cmd
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'state': self.state,
            'has_audio': self.context.raw_audio is not None,
            'has_text': self.context.transcribed_text is not None,
            'has_command': self.context.parsed_command is not None,
            'is_valid': self.context.is_valid,
            'timestamp': self.context.timestamp
        }


def demo_state_machine():
    """Demo state machine functionality"""
    
    # Define callbacks
    def on_listening():
        print("👂 Callback: Start listening for voice input")
    
    def on_processing(context):
        print(f"🔄 Callback: Processing audio (text: {context.transcribed_text})")
    
    def on_command_valid(context):
        print(f"✓ Callback: Valid command: {context.parsed_command}")
    
    def on_command_invalid(context):
        print(f"❌ Callback: Invalid command: {context.error_message}")
    
    def on_speaking(context):
        if context.is_valid:
            print(f"🔊 Callback: Speaking confirmation")
        else:
            print(f"🔊 Callback: Speaking error and suggestion")
    
    def on_executing(context):
        print(f"🤖 Callback: Executing: {context.parsed_command}")
    
    callbacks = {
        'on_listening': on_listening,
        'on_processing': on_processing,
        'on_command_valid': on_command_valid,
        'on_command_invalid': on_command_invalid,
        'on_speaking': on_speaking,
        'on_executing': on_executing
    }
    
    # Create state machine
    sm = RobotStateMachine(callbacks)
    
    print("\n=== Demo: Successful Command Flow ===\n")
    
    # Simulate successful command
    sm.start_listening()
    time.sleep(0.5)
    
    sm.set_audio(b'fake_audio_data')
    sm.audio_captured()
    time.sleep(0.5)
    
    sm.set_transcription("move left a little")
    sm.transcription_complete()
    time.sleep(0.5)
    
    sm.set_command({'action': 'rotate', 'joint': 'base', 'direction': 'left', 'amount': 15}, is_valid=True)
    sm.command_recognized()
    time.sleep(0.5)
    
    sm.nod_complete()
    time.sleep(0.5)
    
    sm.feedback_complete_valid()
    time.sleep(0.5)
    
    sm.motion_complete()
    time.sleep(0.5)
    
    sm.ready_for_next()
    
    print("\n=== Demo: Failed Command Flow ===\n")
    
    # Reset and simulate failed command
    sm.reset_context()
    
    sm.set_audio(b'fake_audio_data_2')
    sm.audio_captured()
    time.sleep(0.5)
    
    sm.set_transcription("make it fly")
    sm.transcription_complete()
    time.sleep(0.5)
    
    sm.set_command({}, is_valid=False)
    sm.set_error("Command not recognized")
    sm.command_not_recognized()
    time.sleep(0.5)
    
    sm.set_suggestion("Try saying 'move up' or 'please dance'")
    sm.suggestion_ready()
    time.sleep(0.5)
    
    sm.feedback_complete_invalid()
    
    print(f"\n✓ Final state: {sm.state}")


if __name__ == "__main__":
    demo_state_machine()

