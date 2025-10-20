#!/usr/bin/env python3
"""
Gesture library for robot arm.
Defines pre-programmed gesture sequences.
"""

import time
from typing import List, Dict, Any, Callable


class Gesture:
    """Base class for gestures"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.sequence = []
    
    def add_step(self, servo_positions: Dict[int, float], duration: int = 800):
        """
        Add a step to the gesture sequence.
        
        Args:
            servo_positions: Dict of {servo_id: angle}
            duration: Movement duration in milliseconds
        """
        self.sequence.append({
            'positions': servo_positions,
            'duration': duration
        })
    
    def get_sequence(self) -> List[Dict[str, Any]]:
        """Get the gesture sequence"""
        return self.sequence


class GestureLibrary:
    """Library of robot arm gestures"""
    
    def __init__(self, servo_config: Dict = None):
        """
        Initialize gesture library.
        
        Args:
            servo_config: Servo configuration with IDs and limits
        """
        self.servo_config = servo_config or {
            'base': 1,
            'shoulder': 2,
            'elbow': 3,
            'wrist': 4,
            'gripper': 5
        }
        
        # Define all gestures
        self.gestures = {
            'nod': self._create_nod(),
            'shake': self._create_shake(),
            'dance': self._create_dance(),
            'wiggle': self._create_wiggle(),
            'wave': self._create_wave()
        }
        
        print(f"✓ Gesture library initialized with {len(self.gestures)} gestures")
    
    def _create_nod(self) -> Gesture:
        """Create nod gesture (yes)"""
        nod = Gesture('nod', 'Nod up and down to indicate yes')
        
        base = self.servo_config['base']
        wrist = self.servo_config['wrist']
        
        # Nod down
        nod.add_step({wrist: 30}, duration=500)
        nod.add_step({wrist: 30}, duration=300)  # Pause
        
        # Nod up
        nod.add_step({wrist: -20}, duration=500)
        nod.add_step({wrist: -20}, duration=300)  # Pause
        
        # Return to center
        nod.add_step({wrist: 0}, duration=500)
        
        return nod
    
    def _create_shake(self) -> Gesture:
        """Create head shake gesture (no)"""
        shake = Gesture('shake', 'Shake side to side to indicate no')
        
        base = self.servo_config['base']
        
        # Shake sequence
        shake.add_step({base: 20}, duration=400)
        shake.add_step({base: -20}, duration=400)
        shake.add_step({base: 20}, duration=400)
        shake.add_step({base: -20}, duration=400)
        
        # Return to center
        shake.add_step({base: 0}, duration=500)
        
        return shake
    
    def _create_dance(self) -> Gesture:
        """Create dance gesture"""
        dance = Gesture('dance', 'Fun dance sequence')
        
        base = self.servo_config['base']
        shoulder = self.servo_config['shoulder']
        elbow = self.servo_config['elbow']
        wrist = self.servo_config['wrist']
        gripper = self.servo_config['gripper']
        
        # Dance moves!
        
        # 1. Wave arms up
        dance.add_step({
            shoulder: 30,
            elbow: -30,
            wrist: 20
        }, duration=600)
        
        # 2. Rotate base left
        dance.add_step({
            base: -30,
            shoulder: 30,
            elbow: -30
        }, duration=600)
        
        # 3. Rotate base right
        dance.add_step({
            base: 30,
            shoulder: 30,
            elbow: -30
        }, duration=600)
        
        # 4. Back to center with wiggle
        dance.add_step({
            base: 0,
            shoulder: 45,
            elbow: -45
        }, duration=500)
        
        # 5. Open/close gripper
        dance.add_step({gripper: 80}, duration=400)
        dance.add_step({gripper: 0}, duration=400)
        dance.add_step({gripper: 80}, duration=400)
        dance.add_step({gripper: 0}, duration=400)
        
        # 6. Wave goodbye
        dance.add_step({
            wrist: 30
        }, duration=300)
        dance.add_step({
            wrist: -30
        }, duration=300)
        dance.add_step({
            wrist: 30
        }, duration=300)
        dance.add_step({
            wrist: -30
        }, duration=300)
        
        # 7. Return to center
        dance.add_step({
            base: 0,
            shoulder: 0,
            elbow: 0,
            wrist: 0,
            gripper: 0
        }, duration=800)
        
        return dance
    
    def _create_wiggle(self) -> Gesture:
        """Create wiggle gesture"""
        wiggle = Gesture('wiggle', 'Rapid wiggling of all joints')
        
        base = self.servo_config['base']
        shoulder = self.servo_config['shoulder']
        elbow = self.servo_config['elbow']
        wrist = self.servo_config['wrist']
        
        # Quick wiggle sequence
        for _ in range(4):
            # Wiggle right
            wiggle.add_step({
                base: 15,
                shoulder: 15,
                elbow: 15,
                wrist: 15
            }, duration=200)
            
            # Wiggle left
            wiggle.add_step({
                base: -15,
                shoulder: -15,
                elbow: -15,
                wrist: -15
            }, duration=200)
        
        # Return to center
        wiggle.add_step({
            base: 0,
            shoulder: 0,
            elbow: 0,
            wrist: 0
        }, duration=500)
        
        return wiggle
    
    def _create_wave(self) -> Gesture:
        """Create wave gesture"""
        wave = Gesture('wave', 'Wave hello')
        
        base = self.servo_config['base']
        shoulder = self.servo_config['shoulder']
        elbow = self.servo_config['elbow']
        wrist = self.servo_config['wrist']
        
        # Raise arm
        wave.add_step({
            shoulder: 45,
            elbow: -30
        }, duration=600)
        
        # Wave wrist
        for _ in range(3):
            wave.add_step({wrist: 30}, duration=300)
            wave.add_step({wrist: -30}, duration=300)
        
        # Lower arm
        wave.add_step({
            shoulder: 0,
            elbow: 0,
            wrist: 0
        }, duration=600)
        
        return wave
    
    def get_gesture(self, name: str) -> Gesture:
        """
        Get gesture by name.
        
        Args:
            name: Gesture name
            
        Returns:
            Gesture object
            
        Raises:
            KeyError if gesture not found
        """
        if name not in self.gestures:
            raise KeyError(f"Gesture '{name}' not found. Available: {list(self.gestures.keys())}")
        
        return self.gestures[name]
    
    def list_gestures(self) -> List[str]:
        """Get list of available gesture names"""
        return list(self.gestures.keys())
    
    def execute_gesture(self, name: str, move_servo_func: Callable, delay_func: Callable = time.sleep):
        """
        Execute a gesture using provided servo control function.
        
        Args:
            name: Gesture name
            move_servo_func: Function(servo_id, angle, duration) to move servos
            delay_func: Function(seconds) to delay between steps
        """
        gesture = self.get_gesture(name)
        
        print(f"🎭 Executing gesture: {gesture.name}")
        
        for i, step in enumerate(gesture.sequence):
            # Move all servos in this step
            for servo_id, angle in step['positions'].items():
                move_servo_func(servo_id, angle, step['duration'])
            
            # Wait for movement to complete
            delay_func(step['duration'] / 1000.0)
        
        print(f"✓ Gesture '{gesture.name}' complete")


def demo_gestures():
    """Demo gesture library"""
    library = GestureLibrary()
    
    print("=== Gesture Library Demo ===\n")
    print(f"Available gestures: {library.list_gestures()}\n")
    
    # Show gesture sequences
    for name in library.list_gestures():
        gesture = library.get_gesture(name)
        print(f"\n{gesture.name.upper()}: {gesture.description}")
        print(f"Steps: {len(gesture.sequence)}")
        
        for i, step in enumerate(gesture.sequence, 1):
            print(f"  {i}. Move servos {step['positions']} over {step['duration']}ms")


if __name__ == "__main__":
    demo_gestures()

