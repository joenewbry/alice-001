#!/usr/bin/env python3
"""
Motion primitives for robot arm control.
Provides high-level motion commands.
"""

from typing import Dict, Optional
import time


class MotionController:
    """High-level motion control for robot arm"""
    
    def __init__(self, robot_controller, config: Dict = None):
        """
        Initialize motion controller.
        
        Args:
            robot_controller: Low-level robot controller (HiwonderS1Controller)
            config: Robot configuration
        """
        self.robot = robot_controller
        self.config = config or {}
        
        # Get motion amounts from config
        motion_config = self.config.get('motion', {})
        self.motion_amounts = {
            'small': motion_config.get('small', 15),
            'medium': motion_config.get('medium', 45),
            'large': motion_config.get('large', 75)
        }
        
        self.default_speed = motion_config.get('default_speed', 800)
        
        # Joint mapping
        self.joints = self.config.get('servos', {})
        
        print("✓ Motion controller initialized")
    
    def move_joint(self, joint_name: str, direction: str, amount: int, speed: Optional[int] = None):
        """
        Move a specific joint.
        
        Args:
            joint_name: Name of joint (base, shoulder, elbow, wrist)
            direction: Direction (left, right, up, down, forward, back)
            amount: Motion amount in degrees
            speed: Motion speed in milliseconds (optional)
        """
        if joint_name not in self.joints:
            print(f"❌ Unknown joint: {joint_name}")
            return False
        
        joint_config = self.joints[joint_name]
        servo_id = joint_config['id']
        
        # Get current position (or use center as default)
        current_pos = self.robot.current_positions.get(servo_id, joint_config.get('center', 0))
        
        # Calculate direction multiplier
        direction_map = {
            'left': -1,
            'right': 1,
            'up': 1,
            'down': -1,
            'forward': 1,
            'back': -1
        }
        
        multiplier = direction_map.get(direction, 1)
        
        # Calculate new position
        new_pos = current_pos + (amount * multiplier)
        
        # Clamp to joint limits
        new_pos = max(joint_config['min_angle'], min(joint_config['max_angle'], new_pos))
        
        # Move servo
        speed = speed or self.default_speed
        return self.robot.move_servo(servo_id, new_pos, speed)
    
    def move_directional(self, direction: str, amount: int, speed: Optional[int] = None):
        """
        Move in a direction (simplified interface).
        
        Args:
            direction: Direction (left, right, up, down, forward, back)
            amount: Motion amount in degrees
            speed: Motion speed in milliseconds
        """
        # Map directions to joints
        joint_map = {
            'left': 'base',
            'right': 'base',
            'up': 'shoulder',
            'down': 'shoulder',
            'forward': 'elbow',
            'back': 'elbow'
        }
        
        joint = joint_map.get(direction)
        if not joint:
            print(f"❌ Unknown direction: {direction}")
            return False
        
        return self.move_joint(joint, direction, amount, speed)
    
    def grasp(self, close_amount: int = 80):
        """Close gripper to grasp"""
        if 'gripper' not in self.joints:
            print("❌ No gripper configured")
            return False
        
        gripper_config = self.joints['gripper']
        return self.robot.move_servo(
            gripper_config['id'],
            close_amount,
            self.default_speed
        )
    
    def release(self):
        """Open gripper to release"""
        if 'gripper' not in self.joints:
            print("❌ No gripper configured")
            return False
        
        gripper_config = self.joints['gripper']
        return self.robot.move_servo(
            gripper_config['id'],
            0,
            self.default_speed
        )
    
    def center_all(self):
        """Move all joints to center position"""
        print("🏠 Centering all joints")
        success = True
        
        for joint_name, joint_config in self.joints.items():
            center_pos = joint_config.get('center', 0)
            result = self.robot.move_servo(
                joint_config['id'],
                center_pos,
                self.default_speed * 2  # Slower for safety
            )
            success = success and result
            time.sleep(0.1)  # Slight delay between joints
        
        return success
    
    def emergency_stop(self):
        """Emergency stop all motion"""
        print("🛑 Emergency stop")
        self.robot.emergency_stop()
    
    def get_joint_position(self, joint_name: str) -> Optional[float]:
        """Get current position of a joint"""
        if joint_name not in self.joints:
            return None
        
        servo_id = self.joints[joint_name]['id']
        return self.robot.current_positions.get(servo_id, 0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """Get all joint positions"""
        positions = {}
        for joint_name in self.joints:
            pos = self.get_joint_position(joint_name)
            if pos is not None:
                positions[joint_name] = pos
        return positions


def demo_motion():
    """Demo motion controller"""
    # This is a simplified demo - in real use, pass actual HiwonderS1Controller
    print("=== Motion Controller Demo ===")
    print("(This would control actual robot hardware)")
    
    # Example command sequence
    commands = [
        ("Move left 15°", "move_directional", ("left", 15)),
        ("Move up 45°", "move_directional", ("up", 45)),
        ("Grasp", "grasp", ()),
        ("Move down 30°", "move_directional", ("down", 30)),
        ("Release", "release", ()),
        ("Center", "center_all", ())
    ]
    
    for desc, method, args in commands:
        print(f"\n{desc}: {method}{args}")


if __name__ == "__main__":
    demo_motion()

