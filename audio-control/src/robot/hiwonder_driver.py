#!/usr/bin/env python3
"""
Extended Hiwonder S1 Robot Controller with gesture support.
Integrates the original controller with gesture library and motion primitives.
"""

import time
import sys
from typing import Dict, Optional
from pathlib import Path

# Try to import xarm library
try:
    import xarm
    XARM_AVAILABLE = True
except ImportError:
    XARM_AVAILABLE = False
    print("⚠️ xarm library not found - using simulation mode")

from .gestures import GestureLibrary
from .motion import MotionController


class HiwonderS1Controller:
    """
    Enhanced Hiwonder S1 controller with gesture support.
    Based on the original HiwonderS1Controller but extended for voice control.
    """
    
    def __init__(self, config: Dict, port: str = None):
        """
        Initialize controller.
        
        Args:
            config: Robot configuration from robot_config.yaml
            port: Serial port for robot connection
        """
        self.config = config
        self.servos = config.get('servos', {})
        self.motion_config = config.get('motion', {})
        self.safety_config = config.get('safety', {})
        
        # Robot state
        self.current_positions = {}
        self.connected = False
        self.arm = None
        self.port = port or '/dev/tty.usbmodemSN234567892'
        
        # Initialize positions to center
        for joint_name, joint_config in self.servos.items():
            servo_id = joint_config['id']
            center = joint_config.get('center', 0)
            self.current_positions[servo_id] = center
        
        # Initialize connection
        self._initialize_connection()
        
        # Create gesture library
        servo_ids = {name: cfg['id'] for name, cfg in self.servos.items()}
        self.gestures = GestureLibrary(servo_ids)
        
        # Create motion controller
        self.motion = MotionController(self, config)
        
        print("✓ Enhanced Hiwonder S1 Controller initialized")
    
    def _initialize_connection(self):
        """Initialize connection to Hiwonder S1"""
        if XARM_AVAILABLE:
            try:
                print(f"🔌 Attempting to connect via USB...")
                self.arm = xarm.Controller('USB')
                self.connected = True
                print("✓ Connected to Hiwonder S1 via USB")
                
                # Get battery voltage
                try:
                    voltage = self.arm.getBatteryVoltage()
                    print(f"🔋 Battery voltage: {voltage:.2f}V")
                except Exception as e:
                    print(f"⚠️ Could not read battery voltage: {e}")
                    
            except Exception as e:
                print(f"⚠️ Failed to connect via USB: {e}")
                print("🔧 Running in simulation mode")
                self.connected = False
        else:
            print("🔧 Running in simulation mode (no hardware)")
            self.connected = False
    
    def move_servo(self, servo_id: int, angle: float, move_time: int = None) -> bool:
        """
        Move a servo to specified angle.
        
        Args:
            servo_id: Servo ID
            angle: Target angle in degrees
            move_time: Movement time in milliseconds
            
        Returns:
            True if successful
        """
        if move_time is None:
            move_time = self.motion_config.get('default_speed', 800)
        
        # Convert angle to servo position
        servo_position = self._angle_to_servo_position(angle, servo_id)
        
        if XARM_AVAILABLE and self.connected and self.arm:
            try:
                self.arm.setPosition(servo_id, servo_position, move_time, wait=False)
                print(f"🤖 USB: Servo {servo_id} -> {angle:.1f}° (time: {move_time}ms)")
                self.current_positions[servo_id] = angle
                return True
            except Exception as e:
                print(f"❌ Servo move failed: {e}")
                return False
        else:
            # Simulation mode
            print(f"🎮 SIM: Servo {servo_id} -> {angle:.1f}° (time: {move_time}ms)")
            self.current_positions[servo_id] = angle
            return True
    
    def _angle_to_servo_position(self, angle: float, servo_id: int) -> int:
        """Convert angle to servo position value (0-1000 range)"""
        # Find servo config
        for joint_name, joint_config in self.servos.items():
            if joint_config['id'] == servo_id:
                min_angle = joint_config['min_angle']
                max_angle = joint_config['max_angle']
                
                # Map angle to 0-1000 range
                angle_range = max_angle - min_angle
                normalized = (angle - min_angle) / angle_range
                position = int(normalized * 1000)
                return max(0, min(1000, position))
        
        # Default mapping
        normalized = (angle + 90) / 180
        position = int(normalized * 1000)
        return max(0, min(1000, position))
    
    def perform_gesture(self, gesture_name: str) -> bool:
        """
        Perform a pre-programmed gesture.
        
        Args:
            gesture_name: Name of gesture (nod, shake, dance, wiggle, wave)
            
        Returns:
            True if successful
        """
        try:
            self.gestures.execute_gesture(
                gesture_name,
                move_servo_func=self.move_servo,
                delay_func=time.sleep
            )
            return True
        except Exception as e:
            print(f"❌ Gesture failed: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop - disable all servos"""
        if XARM_AVAILABLE and self.connected and self.arm:
            try:
                self.arm.servoOff()
                print("🛑 Emergency stop - all servos disabled")
            except Exception as e:
                print(f"❌ Emergency stop failed: {e}")
        else:
            print("🛑 SIM: Emergency stop")
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.connected:
            print("📴 Disconnecting from Hiwonder S1")
            self.connected = False


def demo_enhanced_controller():
    """Demo enhanced controller with gestures"""
    import yaml
    
    print("=== Enhanced Hiwonder S1 Demo ===\n")
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'robot_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['robot']
    except:
        # Fallback config
        config = {
            'servos': {
                'base': {'id': 1, 'min_angle': -90, 'max_angle': 90, 'center': 0},
                'shoulder': {'id': 2, 'min_angle': -90, 'max_angle': 90, 'center': 0},
                'elbow': {'id': 3, 'min_angle': -90, 'max_angle': 90, 'center': 0},
                'wrist': {'id': 4, 'min_angle': -90, 'max_angle': 90, 'center': 0},
                'gripper': {'id': 5, 'min_angle': 0, 'max_angle': 100, 'center': 0},
            },
            'motion': {
                'small': 15,
                'medium': 45,
                'large': 75,
                'default_speed': 800
            }
        }
    
    # Create controller
    controller = HiwonderS1Controller(config)
    
    try:
        # Demo motion commands
        print("\n1. Testing motion primitives:")
        controller.motion.center_all()
        time.sleep(1)
        
        controller.motion.move_directional('left', 15)
        time.sleep(1)
        
        controller.motion.move_directional('up', 30)
        time.sleep(1)
        
        controller.motion.grasp()
        time.sleep(1)
        
        controller.motion.release()
        time.sleep(1)
        
        # Demo gestures
        print("\n2. Testing gestures:")
        for gesture_name in ['nod', 'wiggle', 'dance']:
            print(f"\nPerforming: {gesture_name}")
            controller.perform_gesture(gesture_name)
            time.sleep(1)
        
        # Return to center
        print("\n3. Returning to center:")
        controller.motion.center_all()
        
        print("\n✓ Demo complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        controller.emergency_stop()
    finally:
        controller.disconnect()


if __name__ == "__main__":
    demo_enhanced_controller()

