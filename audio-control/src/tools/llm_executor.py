#!/usr/bin/env python3
"""
LLM-based executor using OpenAI function calling to process voice commands.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional


class LLMExecutor:
    """Uses LLM function calling to convert voice commands to tool calls"""
    
    SYSTEM_PROMPT = """You are a robot arm controller. You receive voice commands and must call the appropriate tools to execute them.

Available joints:
- Base (ID 6): Rotates left/right
- Shoulder (ID 5): Moves up/down  
- Elbow (ID 4): Moves forward/back
- Wrist (ID 3): Moves up/down
- Gripper (ID 1): Opens/closes

Key guidelines:
1. Use positive degrees for: right, up, forward
2. Use negative degrees for: left, down, back
3. Default to 45 degrees if amount not specified
4. "a little" = 15 degrees, "a lot" = 75 degrees
5. For gestures, call the perform_* tools
6. You can call multiple tools in sequence for complex commands

Examples:
- "move left" → rotate_base(degrees=-45)
- "rotate base 30 degrees right" → rotate_base(degrees=30)
- "move shoulder up a lot" → move_shoulder(degrees=75)
- "grasp" → close_gripper(amount=80)
- "dance" → perform_dance()
- "move left then grasp" → rotate_base(degrees=-45), then close_gripper()

Always confirm understanding and execute confidently."""
    
    def __init__(self, robot_tools, model: str = "gpt-4o"):
        """
        Initialize LLM executor.
        
        Args:
            robot_tools: RobotTools instance
            model: OpenAI model to use (needs function calling)
        """
        load_dotenv()
        
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            raise ValueError("OPEN_AI_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.robot_tools = robot_tools
        
        # Add introspection tools
        self._add_introspection_tools()
        
        print(f"✓ LLM executor initialized with {model}")
    
    def _add_introspection_tools(self):
        """Add tools for listing tools and queue status"""
        
        # List available tools
        self.robot_tools.register_tool(
            name="list_available_tools",
            description="List all available robot control tools and their descriptions",
            parameters={},
            function=self._list_tools
        )
        
        # Get queue status (will be connected to queue later)
        self.robot_tools.register_tool(
            name="get_queue_status",
            description="Get the current command queue status (how many tasks pending/running)",
            parameters={},
            function=self._get_queue_status
        )
        
        self.command_queue = None  # Will be set later
    
    def set_command_queue(self, command_queue):
        """Set the command queue reference"""
        self.command_queue = command_queue
    
    def _list_tools(self) -> str:
        """List all available tools"""
        tools = self.robot_tools.list_all_tools()
        result = "Available tools:\\n"
        for tool_name in tools:
            tool_info = self.robot_tools.get_tool_info(tool_name)
            result += f"  • {tool_name}: {tool_info['description']}\\n"
        return result
    
    def _get_queue_status(self) -> str:
        """Get queue status"""
        if not self.command_queue:
            return "Queue not available"
        
        status = self.command_queue.get_queue_status()
        tasks = self.command_queue.list_tasks(5)
        
        result = f"Queue Status:\\n"
        result += f"  • Pending: {status['tasks']['pending']}\\n"
        result += f"  • Running: {status['tasks']['running']}\\n"
        result += f"  • Completed: {status['tasks']['completed']}\\n"
        result += f"  • Current: {status['current_task'] or 'None'}\\n"
        
        if tasks:
            result += f"\\nRecent tasks:\\n"
            for task in tasks[:3]:
                result += f"  • {task['id']}: {task['command']} [{task['status']}]\\n"
        
        return result
    
    def process_command(self, command_text: str) -> List[Dict[str, Any]]:
        """
        Process voice command using LLM function calling.
        
        Args:
            command_text: Voice command text
            
        Returns:
            List of tool calls to execute
        """
        try:
            print(f"🤖 LLM processing: \"{command_text}\"")
            
            # Get tools in OpenAI format
            tools = self.robot_tools.get_tools_for_llm()
            
            # Call LLM with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": command_text}
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            message = response.choices[0].message
            
            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    tool_calls.append({
                        "name": function_name,
                        "arguments": arguments
                    })
                    
                    print(f"   → {function_name}({arguments})")
            
            if not tool_calls:
                print("   ⚠️ No tool calls generated")
                return []
            
            return tool_calls
            
        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return []
    
    def process_and_explain(self, command_text: str) -> tuple[List[Dict[str, Any]], str]:
        """
        Process command and get explanation.
        
        Returns:
            (tool_calls, explanation)
        """
        try:
            tools = self.robot_tools.get_tools_for_llm()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": command_text}
                ],
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            message = response.choices[0].message
            
            # Get explanation
            explanation = message.content or "Executing command"
            
            # Get tool calls
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })
            
            return tool_calls, explanation
            
        except Exception as e:
            return [], f"Error: {e}"


def demo_executor():
    """Demo LLM executor"""
    print("=== LLM Executor Demo ===\\n")
    
    # This would use actual robot tools
    from robot_tools import RobotTools
    
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
            print(f"    [Mock] Moving servo {servo_id} to {angle}°")
        
        def perform_gesture(self, name):
            print(f"    [Mock] Performing {name}")
        
        class motion:
            @staticmethod
            def center_all():
                print("    [Mock] Centering")
        
        def emergency_stop(self):
            print("    [Mock] Emergency stop")
    
    mock_robot = MockRobot()
    tools = RobotTools(mock_robot)
    executor = LLMExecutor(tools)
    
    test_commands = [
        "rotate base 45 degrees left",
        "move shoulder up a lot",
        "grasp the object",
        "please dance",
        "move left then move up"
    ]
    
    for cmd in test_commands:
        print(f"\\nCommand: \"{cmd}\"")
        tool_calls = executor.process_command(cmd)
        
        if tool_calls:
            print(f"  ✓ Generated {len(tool_calls)} tool call(s)")
        else:
            print(f"  ❌ No tools generated")


if __name__ == "__main__":
    demo_executor()

