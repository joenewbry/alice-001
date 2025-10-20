#!/usr/bin/env python3
"""
Tool-based voice control system with LLM function calling and command queue.
This is the enhanced version with queue support.
"""

import sys
import yaml
import time
from pathlib import Path
from typing import Dict, Any

# Import all modules
from audio import AudioCapture, FasterWhisperSTT, TextToSpeech
from robot import HiwonderS1Controller
from tools import RobotTools, CommandQueue, LLMExecutor
from command_reference import print_command_reference


class ToolBasedVoiceControl:
    """Tool-based voice control with queue support"""
    
    def __init__(self, config_dir: Path = None):
        """Initialize the tool-based control system"""
        print("=" * 60)
        print("🔧 TOOL-BASED VOICE CONTROL SYSTEM")
        print("=" * 60)
        
        # Load configurations
        self.config_dir = config_dir or Path(__file__).parent.parent / 'config'
        self.configs = self._load_configs()
        
        # Initialize components
        print("\n📦 Initializing components...\n")
        
        # 1. Audio system
        self.audio_capture = AudioCapture(self.configs['audio'])
        self.speech_to_text = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
        self.text_to_speech = TextToSpeech(voice="alloy", model="tts-1")  # tts-1 is faster
        
        # 2. Robot controller
        self.robot = HiwonderS1Controller(self.configs['robot'])
        
        # 3. Tool system
        self.robot_tools = RobotTools(self.robot)
        
        # 4. Command queue
        self.command_queue = CommandQueue(self.robot_tools, max_queue_size=10)
        
        # 5. LLM executor (uses gpt-4o-mini by default for speed)
        self.llm_executor = LLMExecutor(self.robot_tools)
        self.llm_executor.set_command_queue(self.command_queue)
        
        # State
        self.running = False
        
        print("\n✓ All components initialized successfully!")
        print("\n" + "=" * 60 + "\n")
        
        # Display command reference
        print_command_reference()
        
        # Show available tools
        self._print_available_tools()
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files"""
        configs = {}
        
        config_files = {
            'audio': 'audio_config.yaml',
            'robot': 'robot_config.yaml'
        }
        
        for key, filename in config_files.items():
            config_path = self.config_dir / filename
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    configs[key] = data.get(key, data)
                    print(f"✓ Loaded {filename}")
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
                configs[key] = {}
        
        return configs
    
    def _print_available_tools(self):
        """Print all available tools"""
        print("\n🔧 AVAILABLE TOOLS:")
        print("=" * 60)
        tools = self.robot_tools.list_all_tools()
        
        categories = {
            "Joint Control": ["rotate_base", "move_shoulder", "move_elbow", "move_wrist"],
            "Gripper": ["open_gripper", "close_gripper"],
            "Gestures": ["perform_dance", "perform_wiggle", "perform_nod", "perform_wave"],
            "System": ["center_robot", "emergency_stop"],
            "Info": ["list_available_tools", "get_queue_status"]
        }
        
        for category, tool_names in categories.items():
            print(f"\n{category}:")
            for tool_name in tool_names:
                if tool_name in tools:
                    tool_info = self.robot_tools.get_tool_info(tool_name)
                    print(f"  • {tool_name}: {tool_info['description']}")
        
        print("\n" + "=" * 60)
    
    def run(self):
        """Run the main control loop"""
        self.running = True
        
        print("\n🎤 TOOL-BASED VOICE CONTROL READY!")
        print("   Features:")
        print("   • LLM function calling for intelligent command processing")
        print("   • Command queue - say multiple commands, they'll execute in order")
        print("   • Tool-based architecture - each joint is a separate tool")
        print("   • Say 'list tools' to see all available tools")
        print("   • Say 'queue status' to check command queue\n")
        
        try:
            # Start the command queue
            self.command_queue.start()
            
            # Center robot at start
            print("🏠 Centering robot...")
            self.robot.motion.center_all()
            time.sleep(1)
            
            # Greeting (non-blocking for faster startup)
            self.text_to_speech.speak("Hello! Tool-based voice control is ready. You can queue multiple commands.", blocking=False)
            
            # Main loop
            while self.running:
                should_continue = self.run_once()
                if not should_continue:
                    break
                
                # Brief pause between listening cycles
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
        
        finally:
            self.cleanup()
    
    def run_once(self) -> bool:
        """Process one voice command"""
        try:
            print("\n👂 Listening for voice command...")
            print("   (Speak now or press Ctrl+C to exit)")
            
            # Start timing for latency tracking
            import time as time_module
            start_time = time_module.time()
            
            # Capture audio
            audio_data = self.audio_capture.record_utterance()
            
            if not audio_data:
                print("⚠️ No audio captured")
                return True
            
            capture_time = time_module.time()
            
            # Transcribe
            print("🔄 Transcribing...")
            text = self.speech_to_text.transcribe(audio_data, self.configs['audio']['sample_rate'])
            
            if not text:
                print("⚠️ Could not transcribe audio")
                return True
            
            transcribe_time = time_module.time()
            print(f"📝 You said: \"{text}\"")
            
            # Process with LLM to get tool calls
            tool_calls, explanation = self.llm_executor.process_and_explain(text)
            
            llm_time = time_module.time()
            
            if not tool_calls:
                self.text_to_speech.speak("I'm not sure what you want me to do. Try saying 'list tools' to see what I can do.", stream=True)
                return True
            
            # Add to queue (robot starts executing immediately in background)
            task_id = self.command_queue.add_task(text, tool_calls)
            
            # Confirm with streaming (lower latency, non-blocking so robot can move in parallel)
            queue_size = self.command_queue.get_queue_status()['queue_size']
            if queue_size > 1:
                self.text_to_speech.speak(f"{explanation}. Added to queue, {queue_size} commands pending.", blocking=False, stream=True)
            else:
                self.text_to_speech.speak(f"{explanation}", blocking=False, stream=True)
            
            tts_start_time = time_module.time()
            
            # Print latency breakdown
            print(f"\n⏱️  LATENCY BREAKDOWN:")
            print(f"   • Audio capture: {(capture_time - start_time):.2f}s")
            print(f"   • STT (faster-whisper): {(transcribe_time - capture_time):.2f}s")
            print(f"   • LLM (gpt-4o-mini): {(llm_time - transcribe_time):.2f}s")
            print(f"   • TTS start (streaming): {(tts_start_time - llm_time):.2f}s")
            print(f"   📊 Total to audio output: {(tts_start_time - start_time):.2f}s")
            print(f"   🎯 Target: < 2.0s")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return True
    
    def cleanup(self):
        """Cleanup all resources"""
        print("\n🧹 Cleaning up...")
        
        # Stop the queue
        self.command_queue.stop()
        
        # Cancel pending tasks
        cancelled = self.command_queue.cancel_all()
        if cancelled > 0:
            print(f"🚫 Cancelled {cancelled} pending tasks")
        
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
        print("\nThank you for using the tool-based voice control system!")


def main():
    """Main entry point"""
    try:
        # Create and run application
        app = ToolBasedVoiceControl()
        app.run()
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

