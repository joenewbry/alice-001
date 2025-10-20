#!/usr/bin/env python3
"""
Command queue system for handling multiple voice commands.
Allows commands to be queued while others are executing.
"""

import threading
import queue
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a queued task"""
    id: str
    command_text: str
    tool_calls: List[Dict[str, Any]]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class CommandQueue:
    """Queue-based command execution system"""
    
    def __init__(self, robot_tools, max_queue_size: int = 10):
        """
        Initialize command queue.
        
        Args:
            robot_tools: RobotTools instance
            max_queue_size: Maximum number of queued commands
        """
        self.robot_tools = robot_tools
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        self.task_lock = threading.Lock()
        
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.current_task: Optional[Task] = None
        
        print("✓ Command queue initialized")
    
    def start(self):
        """Start the queue processor"""
        if self.running:
            print("⚠️ Queue already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print("🚀 Command queue started")
    
    def stop(self):
        """Stop the queue processor"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("⏹️  Command queue stopped")
    
    def add_task(self, command_text: str, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Add a new task to the queue.
        
        Args:
            command_text: Original voice command
            tool_calls: List of tool calls to execute
            
        Returns:
            Task ID
        """
        with self.task_lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            
            task = Task(
                id=task_id,
                command_text=command_text,
                tool_calls=tool_calls,
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.tasks[task_id] = task
            self.task_queue.put(task)
            
            queue_size = self.task_queue.qsize()
            print(f"➕ Added task {task_id}: '{command_text}' (Queue: {queue_size})")
            
            return task_id
    
    def _process_queue(self):
        """Worker thread that processes queued tasks"""
        print("🔄 Queue processor started")
        
        while self.running:
            try:
                # Get next task (with timeout to check running flag)
                task = self.task_queue.get(timeout=0.5)
                
                # Execute the task
                self._execute_task(task)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Queue processor error: {e}")
    
    def _execute_task(self, task: Task):
        """Execute a single task"""
        with self.task_lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.current_task = task
        
        print(f"🔧 Executing task {task.id}: '{task.command_text}'")
        
        results = []
        
        try:
            # Execute each tool call in sequence
            for tool_call in task.tool_calls:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                print(f"   → Calling {tool_name}({arguments})")
                
                result = self.robot_tools.execute_tool(tool_name, arguments)
                results.append(result)
                
                if not result["success"]:
                    print(f"   ❌ Tool call failed: {result.get('error')}")
                    raise Exception(f"Tool {tool_name} failed: {result.get('error')}")
                
                print(f"   ✓ {result.get('result', 'Done')}")
                
                # Small delay between tool calls
                time.sleep(0.2)
            
            # Task completed successfully
            with self.task_lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = results
                self.current_task = None
            
            print(f"✅ Task {task.id} completed")
            
        except Exception as e:
            # Task failed
            with self.task_lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = str(e)
                self.current_task = None
            
            print(f"❌ Task {task.id} failed: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self.task_lock:
            return {
                "queue_size": self.task_queue.qsize(),
                "total_tasks": len(self.tasks),
                "current_task": self.current_task.id if self.current_task else None,
                "tasks": {
                    "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                    "running": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                    "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                    "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
                }
            }
    
    def list_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent tasks"""
        with self.task_lock:
            tasks = sorted(self.tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]
            return [
                {
                    "id": t.id,
                    "command": t.command_text,
                    "status": t.status.value,
                    "created": t.created_at.strftime("%H:%M:%S"),
                    "duration": (t.completed_at - t.started_at).total_seconds() if t.completed_at and t.started_at else None
                }
                for t in tasks
            ]
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def cancel_all(self):
        """Cancel all pending tasks"""
        with self.task_lock:
            cancelled_count = 0
            while not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    task.status = TaskStatus.CANCELLED
                    cancelled_count += 1
                except queue.Empty:
                    break
            
            print(f"🚫 Cancelled {cancelled_count} pending tasks")
            return cancelled_count


def demo_queue():
    """Demo command queue"""
    print("=== Command Queue Demo ===\n")
    
    # Create mock robot tools
    class MockTools:
        def execute_tool(self, name, args):
            time.sleep(0.5)  # Simulate work
            return {"success": True, "result": f"Executed {name}"}
    
    mock_tools = MockTools()
    cmd_queue = CommandQueue(mock_tools, max_queue_size=5)
    
    # Start queue
    cmd_queue.start()
    
    # Add some tasks
    cmd_queue.add_task("move left", [{"name": "rotate_base", "arguments": {"degrees": -45}}])
    cmd_queue.add_task("move up", [{"name": "move_shoulder", "arguments": {"degrees": 30}}])
    cmd_queue.add_task("dance", [{"name": "perform_dance", "arguments": {}}])
    
    # Check status
    print("\n📊 Queue Status:")
    status = cmd_queue.get_queue_status()
    print(f"   Queue size: {status['queue_size']}")
    print(f"   Current task: {status['current_task']}")
    
    # Wait for completion
    time.sleep(3)
    
    # List tasks
    print("\n📋 Recent Tasks:")
    for task in cmd_queue.list_tasks():
        print(f"   {task['id']}: {task['command']} - {task['status']}")
    
    cmd_queue.stop()


if __name__ == "__main__":
    demo_queue()

