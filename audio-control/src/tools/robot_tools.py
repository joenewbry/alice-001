#!/usr/bin/env python3
"""
Robot tools - Each joint and action as a separate tool for LLM function calling.
"""

from typing import Dict, Any, List, Optional, Callable
import time


class RobotTools:
    """Tool-based interface for robot control"""
    
    def __init__(self, robot_controller):
        """Initialize robot tools with controller"""
        self.robot = robot_controller
        self.tools_registry: Dict[str, Dict[str, Any]] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available tools"""
        
        # Base joint tools
        self.register_tool(
            name="rotate_base",
            description="Rotate the base joint left or right by specified degrees",
            parameters={
                "degrees": {"type": "number", "description": "Degrees to rotate (positive=right, negative=left)"},
            },
            function=self._rotate_base
        )
        
        # Shoulder joint tools
        self.register_tool(
            name="move_shoulder",
            description="Move the shoulder joint up or down by specified degrees",
            parameters={
                "degrees": {"type": "number", "description": "Degrees to move (positive=up, negative=down)"},
            },
            function=self._move_shoulder
        )
        
        # Elbow joint tools
        self.register_tool(
            name="move_elbow",
            description="Move the elbow joint forward or back by specified degrees",
            parameters={
                "degrees": {"type": "number", "description": "Degrees to move (positive=forward, negative=back)"},
            },
            function=self._move_elbow
        )
        
        # Wrist joint tools
        self.register_tool(
            name="move_wrist",
            description="Move the wrist joint up or down by specified degrees",
            parameters={
                "degrees": {"type": "number", "description": "Degrees to move (positive=up, negative=down)"},
            },
            function=self._move_wrist
        )
        
        # Gripper tools
        self.register_tool(
            name="open_gripper",
            description="Open the gripper to release objects",
            parameters={},
            function=self._open_gripper
        )
        
        self.register_tool(
            name="close_gripper",
            description="Close the gripper to grasp objects",
            parameters={
                "amount": {"type": "number", "description": "Amount to close (0-100), default 80", "optional": True}
            },
            function=self._close_gripper
        )
        
        # Gesture tools
        self.register_tool(
            name="perform_dance",
            description="Perform a fun dance sequence with all joints",
            parameters={},
            function=self._perform_dance
        )
        
        self.register_tool(
            name="perform_wiggle",
            description="Wiggle all joints rapidly",
            parameters={},
            function=self._perform_wiggle
        )
        
        self.register_tool(
            name="perform_nod",
            description="Nod the wrist up and down to indicate yes",
            parameters={},
            function=self._perform_nod
        )
        
        self.register_tool(
            name="perform_wave",
            description="Wave hello",
            parameters={},
            function=self._perform_wave
        )
        
        # System tools
        self.register_tool(
            name="center_robot",
            description="Return all joints to the home/center position",
            parameters={},
            function=self._center_robot
        )
        
        self.register_tool(
            name="emergency_stop",
            description="Immediately stop all robot motion",
            parameters={},
            function=self._emergency_stop
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        """Register a new tool"""
        self.tools_registry[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format"""
        tools = []
        for tool_name, tool_info in self.tools_registry.items():
            # Convert to OpenAI format
            properties = {}
            required = []
            
            for param_name, param_info in tool_info["parameters"].items():
                properties[param_name] = {
                    "type": param_info["type"],
                    "description": param_info["description"]
                }
                if not param_info.get("optional", False):
                    required.append(param_name)
            
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        return tools
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with arguments"""
        if tool_name not in self.tools_registry:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        try:
            tool = self.tools_registry[tool_name]
            result = tool["function"](**arguments)
            return {"success": True, "result": result, "tool": tool_name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}
    
    def list_all_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools_registry.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        return self.tools_registry.get(tool_name)
    
    # Tool implementations
    
    def _rotate_base(self, degrees: float) -> str:
        """Rotate base joint"""
        servo_id = self.robot.config['servos']['base']['id']
        current = self.robot.current_positions.get(servo_id, 0)
        new_pos = current + degrees
        self.robot.move_servo(servo_id, new_pos)
        return f"Rotated base {degrees}° (now at {new_pos}°)"
    
    def _move_shoulder(self, degrees: float) -> str:
        """Move shoulder joint"""
        servo_id = self.robot.config['servos']['shoulder']['id']
        current = self.robot.current_positions.get(servo_id, -45)
        new_pos = current + degrees
        self.robot.move_servo(servo_id, new_pos)
        return f"Moved shoulder {degrees}° (now at {new_pos}°)"
    
    def _move_elbow(self, degrees: float) -> str:
        """Move elbow joint"""
        servo_id = self.robot.config['servos']['elbow']['id']
        current = self.robot.current_positions.get(servo_id, -60)
        new_pos = current + degrees
        self.robot.move_servo(servo_id, new_pos)
        return f"Moved elbow {degrees}° (now at {new_pos}°)"
    
    def _move_wrist(self, degrees: float) -> str:
        """Move wrist joint"""
        servo_id = self.robot.config['servos']['wrist']['id']
        current = self.robot.current_positions.get(servo_id, 0)
        new_pos = current + degrees
        self.robot.move_servo(servo_id, new_pos)
        return f"Moved wrist {degrees}° (now at {new_pos}°)"
    
    def _open_gripper(self) -> str:
        """Open gripper"""
        servo_id = self.robot.config['servos']['gripper']['id']
        self.robot.move_servo(servo_id, 0)
        return "Opened gripper"
    
    def _close_gripper(self, amount: float = 80) -> str:
        """Close gripper"""
        servo_id = self.robot.config['servos']['gripper']['id']
        self.robot.move_servo(servo_id, amount)
        return f"Closed gripper to {amount}"
    
    def _perform_dance(self) -> str:
        """Perform dance gesture"""
        self.robot.perform_gesture('dance')
        return "Performed dance"
    
    def _perform_wiggle(self) -> str:
        """Perform wiggle gesture"""
        self.robot.perform_gesture('wiggle')
        return "Performed wiggle"
    
    def _perform_nod(self) -> str:
        """Perform nod gesture"""
        self.robot.perform_gesture('nod')
        return "Performed nod"
    
    def _perform_wave(self) -> str:
        """Perform wave gesture"""
        self.robot.perform_gesture('wave')
        return "Performed wave"
    
    def _center_robot(self) -> str:
        """Center all joints"""
        self.robot.motion.center_all()
        return "Centered robot to home position"
    
    def _emergency_stop(self) -> str:
        """Emergency stop"""
        self.robot.emergency_stop()
        return "Emergency stop activated"


def demo_tools():
    """Demo robot tools"""
    print("=== Robot Tools Demo ===\n")
    
    # This would normally use actual robot controller
    class MockRobot:
        def __init__(self):
            self.config = {
                'servos': {
                    'base': {'id': 6},
                    'shoulder': {'id': 5},
                    'elbow': {'id': 4},
                    'wrist': {'id': 3},
                    'gripper': {'id': 1}
                }
            }
            self.current_positions = {}
        
        def move_servo(self, servo_id, angle):
            print(f"  Moving servo {servo_id} to {angle}°")
        
        def perform_gesture(self, name):
            print(f"  Performing gesture: {name}")
        
        class motion:
            @staticmethod
            def center_all():
                print("  Centering all joints")
        
        def emergency_stop(self):
            print("  Emergency stop!")
    
    mock_robot = MockRobot()
    tools = RobotTools(mock_robot)
    
    print(f"Registered {len(tools.list_all_tools())} tools:\n")
    for tool_name in tools.list_all_tools():
        tool_info = tools.get_tool_info(tool_name)
        print(f"  • {tool_name}: {tool_info['description']}")
    
    print("\n=== Testing Tools ===\n")
    
    # Test some tools
    tests = [
        ("rotate_base", {"degrees": 45}),
        ("move_shoulder", {"degrees": 30}),
        ("close_gripper", {"amount": 80}),
        ("perform_dance", {})
    ]
    
    for tool_name, args in tests:
        print(f"Executing: {tool_name}({args})")
        result = tools.execute_tool(tool_name, args)
        print(f"  Result: {result}\n")


if __name__ == "__main__":
    demo_tools()

