#!/usr/bin/env python3
"""
Main application for voice-controlled robot arm.
Integrates all components: audio, state machine, command parsing, LLM, robot control.
"""

import sys
import yaml
import time
from pathlib import Path
from typing import Dict, Any

# Import all modules
from audio import AudioCapture, SpeechToText, TextToSpeech
from control import RobotStateMachine, CommandParser, CommandOntology
from llm import CommandAssistant
from robot import HiwonderS1Controller
from feedback_system import FeedbackSystem
from command_reference import print_command_reference


class VoiceControlledRobotArm:
    """Main application class"""
    
    def __init__(self, config_dir: Path = None):
        """
        Initialize voice-controlled robot arm system.
        
        Args:
            config_dir: Directory containing configuration files
        """
        print("=" * 60)
        print("VOICE-CONTROLLED ROBOT ARM SYSTEM")
        print("=" * 60)
        
        # Load configurations
        self.config_dir = config_dir or Path(__file__).parent.parent / 'config'
        self.configs = self._load_configs()
        
        # Initialize components
        print("\n📦 Initializing components...\n")
        
        # 1. Audio system
        self.audio_capture = AudioCapture(self.configs['audio'])
        self.speech_to_text = SpeechToText(model_name="base")
        self.text_to_speech = TextToSpeech(voice="alloy")
        
        # 2. Command processing
        self.ontology = CommandOntology(self.configs.get('commands'))
        self.command_parser = CommandParser(self.ontology)
        
        # 3. LLM assistant
        self.llm_assistant = CommandAssistant(model="gpt-4o-mini")
        
        # 4. Robot controller
        self.robot = HiwonderS1Controller(self.configs['robot'])
        
        # 5. Feedback system
        self.feedback = FeedbackSystem(self.robot, self.text_to_speech)
        
        # 6. State machine with callbacks
        callbacks = {
            'on_listening': self._on_listening,
            'on_processing': self._on_processing,
            'on_command_valid': self._on_command_valid,
            'on_command_invalid': self._on_command_invalid,
            'on_speaking': self._on_speaking,
            'on_executing': self._on_executing
        }
        self.state_machine = RobotStateMachine(callbacks)
        
        # State
        self.running = False
        
        print("\n✓ All components initialized successfully!")
        print("\n" + "=" * 60 + "\n")
        
        # Display command reference
        print_command_reference()
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        
        config_files = {
            'audio': 'audio_config.yaml',
            'robot': 'robot_config.yaml',
            'commands': 'commands.yaml'
        }
        
        for key, filename in config_files.items():
            config_path = self.config_dir / filename
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    # Extract nested config if needed
                    configs[key] = data.get(key, data)
                    print(f"✓ Loaded {filename}")
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
                configs[key] = {}
        
        return configs
    
    # State machine callbacks
    
    def _on_listening(self):
        """Callback when entering listening state"""
        print("\n👂 Listening for voice command...")
        print("   (Speak now or press Ctrl+C to exit)")
    
    def _on_processing(self, context):
        """Callback when processing audio"""
        print("🔄 Processing speech...")
        
        # Transcribe audio
        text = self.speech_to_text.transcribe(
            context.raw_audio,
            self.configs['audio']['sample_rate']
        )
        
        if text:
            self.state_machine.set_transcription(text)
            self.state_machine.transcription_complete()
        else:
            print("⚠️ Could not transcribe audio")
            self.state_machine.transcription_failed()
    
    def _on_command_valid(self, context):
        """Callback when command is valid"""
        print(f"✓ Valid command recognized")
        
        # Trigger understanding gesture (nod)
        self.state_machine.nod_complete()
    
    def _on_command_invalid(self, context):
        """Callback when command is invalid"""
        print(f"❌ Command not recognized: {context.transcribed_text}")
        
        # Get LLM suggestion
        suggestion = self.llm_assistant.get_suggestion(
            context.transcribed_text,
            self.llm_assistant.get_context_summary()
        )
        
        self.state_machine.set_suggestion(suggestion)
        self.state_machine.suggestion_ready()
    
    def _on_speaking(self, context):
        """Callback when robot speaks"""
        if context.is_valid:
            # Valid command - acknowledge and prepare to execute
            command_dict = self.command_parser.command_to_dict(context.parsed_command)
            response_text = self.command_parser.get_response_text(
                context.parsed_command,
                True
            )
            
            # Nod yes and speak
            self.feedback.acknowledge_valid_command(response_text)
            
            # Transition to executing
            self.state_machine.feedback_complete_valid()
            
        else:
            # Invalid command - shake no and speak suggestion
            self.feedback.acknowledge_invalid_command(
                context.error_message or "Command not understood",
                context.llm_suggestion
            )
            
            # Add to LLM context
            self.llm_assistant.add_to_context(
                context.transcribed_text or "unknown",
                "failed"
            )
            
            # Return to listening
            self.state_machine.feedback_complete_invalid()
    
    def _on_executing(self, context):
        """Callback when executing command"""
        print(f"🤖 Executing command...")
        
        # Execute the command
        command_dict = self.command_parser.command_to_dict(context.parsed_command)
        success = self.feedback._execute_robot_command(command_dict)
        
        if success:
            print("✓ Command executed successfully")
            
            # Add to LLM context
            self.llm_assistant.add_to_context(
                context.transcribed_text,
                "success"
            )
            
            self.state_machine.motion_complete()
        else:
            print("❌ Command execution failed")
            self.state_machine.motion_failed()
    
    def run_once(self) -> bool:
        """
        Run one command cycle.
        
        Returns:
            True if should continue, False to exit
        """
        # Start listening (only if not already listening)
        if self.state_machine.state != 'listening':
            self.state_machine.start_listening()
        
        # Capture audio
        try:
            audio_data = self.audio_capture.record_utterance()
            
            if not audio_data:
                print("⚠️ No audio captured")
                self.state_machine.timeout()
                return True
            
            # Process audio
            self.state_machine.set_audio(audio_data)
            self.state_machine.audio_captured()
            
            # Wait for processing to complete
            time.sleep(0.5)
            
            # Parse command
            context = self.state_machine.get_context()
            if context.transcribed_text:
                command, is_valid, error = self.command_parser.parse(context.transcribed_text)
                
                if is_valid and command:
                    self.state_machine.set_command(
                        self.command_parser.command_to_dict(command),
                        True
                    )
                    context.parsed_command = command
                    self.state_machine.command_recognized()
                else:
                    self.state_machine.set_command({}, False)
                    self.state_machine.set_error(error or "Unknown error")
                    self.state_machine.command_not_recognized()
            
            # Wait for execution to complete
            while self.state_machine.state not in ['idle', 'listening', 'execution_complete']:
                time.sleep(0.5)
            
            # Ready for next command
            if self.state_machine.state == 'execution_complete':
                self.state_machine.ready_for_next()
                self.state_machine.reset_context()
            
            # Ensure we return to idle before next cycle
            if self.state_machine.state == 'listening':
                self.state_machine.timeout()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            # Emergency stop the state machine
            try:
                self.state_machine.emergency_stop()
            except:
                pass
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover to idle state
            try:
                if self.state_machine.state != 'idle':
                    self.state_machine.emergency_stop()
            except:
                pass
            return True
    
    def run(self):
        """Run the main control loop"""
        self.running = True
        
        print("\n🎤 VOICE CONTROL SYSTEM READY!")
        print("   Try saying:")
        print("   • 'rotate base 45 degrees left' (precise control)")
        print("   • 'move left' (simple command)")
        print("   • 'please dance' (fun gesture)")
        print("   • 'center' (return home)")
        print("   • Say 'stop' anytime to halt\n")
        
        try:
            # Center robot at start
            print("🏠 Centering robot...")
            self.robot.motion.center_all()
            time.sleep(1)
            
            # Greeting
            self.text_to_speech.speak("Hello! I'm ready for voice commands.")
            
            # Main loop
            while self.running:
                should_continue = self.run_once()
                if not should_continue:
                    break
                
                # Brief pause between commands
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\n🧹 Cleaning up...")
        
        try:
            # Say goodbye
            self.text_to_speech.speak("Goodbye!")
            time.sleep(1)
        except:
            pass
        
        # Center robot
        try:
            self.robot.motion.center_all()
            time.sleep(1)
        except:
            pass
        
        # Cleanup resources
        self.audio_capture.cleanup()
        self.text_to_speech.cleanup()
        self.robot.disconnect()
        
        print("✓ Cleanup complete")
        print("\nThank you for using the voice-controlled robot arm system!")


def main():
    """Main entry point"""
    try:
        # Create and run application
        app = VoiceControlledRobotArm()
        app.run()
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

