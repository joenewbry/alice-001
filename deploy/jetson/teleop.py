"""Keyboard teleoperation for collecting demonstrations.

Controls the xArm 1S joints directly via keyboard, with optional recording
for HIL-SERL / Octo fine-tuning demonstrations.

Controls:
    Q/A - Base rotate left/right
    W/S - Shoulder up/down
    E/D - Elbow flex/extend
    R/F - Wrist pitch up/down
    T/G - Wrist roll CW/CCW
    Space - Toggle gripper open/close
    Enter - Mark current episode as success
    Backspace - Mark as failure and restart
    H - Go to home position
    ESC - Quit

Usage:
    python -m deploy.jetson.teleop --record --output /ssd/alice/demos/
"""

import sys
import time
import select
import termios
import tty
from dataclasses import dataclass
from typing import Optional

from .servo_controller import ServoController, ServoConfig
from .data_logger import DataLogger, LogConfig


@dataclass
class TeleopConfig:
    """Teleoperation configuration."""
    step_size_rad: float = 0.05       # Radians per keypress
    gripper_step_rad: float = 0.02    # Gripper step
    control_rate_hz: float = 20.0     # Control loop rate
    record: bool = False
    output_dir: str = "/ssd/alice/demos"


# Key mappings: key -> (joint_index, direction)
KEY_MAP = {
    "q": (0, +1),   # Base left
    "a": (0, -1),   # Base right
    "w": (1, +1),   # Shoulder up
    "s": (1, -1),   # Shoulder down
    "e": (2, +1),   # Elbow flex (toward 0)
    "d": (2, -1),   # Elbow extend (toward -120)
    "r": (3, +1),   # Wrist pitch up
    "f": (3, -1),   # Wrist pitch down
    "t": (4, +1),   # Wrist roll CW
    "g": (4, -1),   # Wrist roll CCW
}


class TeleopController:
    """Keyboard teleoperation with demonstration recording."""

    def __init__(self, config: Optional[TeleopConfig] = None):
        self.config = config or TeleopConfig()
        self.servo = ServoController()
        self.logger = None
        self._gripper_open = True
        self._running = False
        self._episode_count = 0

    def start(self):
        """Start teleoperation (blocking)."""
        self.servo.connect()

        if self.config.record:
            self.logger = DataLogger(LogConfig(log_dir=self.config.output_dir))
            self.logger.open()
            self.logger.new_episode()
            print(f"[Teleop] Recording to {self.config.output_dir}")

        self.servo.go_home(duration_ms=1500)
        time.sleep(1.5)

        self._print_controls()
        self._running = True
        self._run_loop()

    def _run_loop(self):
        """Main teleop loop with non-blocking keyboard input."""
        dt = 1.0 / self.config.control_rate_hz
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while self._running:
                loop_start = time.time()

                # Check for keypress (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    key = sys.stdin.read(1).lower()
                    self._handle_key(key)

                # Log current state
                if self.logger:
                    pos = self.servo.read_positions_radians()
                    self.logger.log_step(
                        joint_pos=pos,
                        actions=[0.0] * 7,  # No policy actions in teleop
                    )

                # Rate limit
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self._shutdown()

    def _handle_key(self, key: str):
        """Process a single keypress."""
        if key == "\x1b":  # ESC
            self._running = False
            return

        if key == "h":
            print("[Teleop] Going home...")
            self.servo.go_home()
            time.sleep(1.0)
            return

        if key == " ":
            self._toggle_gripper()
            return

        if key == "\n" or key == "\r":  # Enter = success
            if self.logger:
                self.logger.end_episode(success=True)
                self._episode_count += 1
                print(f"[Teleop] Episode {self._episode_count} marked SUCCESS")
                self.servo.go_home()
                time.sleep(1.0)
                self.logger.new_episode()
            return

        if key == "\x7f":  # Backspace = failure
            if self.logger:
                self.logger.end_episode(success=False)
                self._episode_count += 1
                print(f"[Teleop] Episode {self._episode_count} marked FAILURE, resetting")
                self.servo.go_home()
                time.sleep(1.0)
                self.logger.new_episode()
            return

        # Joint movement
        if key in KEY_MAP:
            joint_idx, direction = KEY_MAP[key]
            current = self.servo.read_positions_radians()
            current[joint_idx] += direction * self.config.step_size_rad
            self.servo.move_to_radians(current)

    def _toggle_gripper(self):
        """Toggle gripper open/close."""
        self._gripper_open = not self._gripper_open
        current = self.servo.read_positions_radians()

        if self._gripper_open:
            current[5] = 0.26    # left_finger open
            current[6] = -0.26   # right_finger open
            print("[Teleop] Gripper OPEN")
        else:
            current[5] = 0.02    # left_finger closed
            current[6] = -0.02   # right_finger closed
            print("[Teleop] Gripper CLOSE")

        self.servo.move_to_radians(current)

    def _shutdown(self):
        """Clean shutdown."""
        self.servo.hold_position()
        if self.logger:
            if self.logger._buf_timestamps:
                self.logger.end_episode(success=False)
            self.logger.close()
        self.servo.disconnect()
        print(f"\n[Teleop] Done. {self._episode_count} episodes recorded.")

    def _print_controls(self):
        """Print control instructions."""
        print(f"\n{'='*50}")
        print("  Alice xArm Teleoperation")
        print(f"{'='*50}")
        print("  Q/A  - Base rotate")
        print("  W/S  - Shoulder")
        print("  E/D  - Elbow")
        print("  R/F  - Wrist pitch")
        print("  T/G  - Wrist roll")
        print("  Space - Toggle gripper")
        print("  Enter - Mark success + new episode")
        print("  Bksp  - Mark failure + new episode")
        print("  H     - Home position")
        print("  ESC   - Quit")
        if self.config.record:
            print(f"\n  Recording to: {self.config.output_dir}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="xArm Teleoperation")
    parser.add_argument("--record", action="store_true", help="Record demonstrations")
    parser.add_argument("--output", type=str, default="/ssd/alice/demos", help="Output directory")
    parser.add_argument("--step-size", type=float, default=0.05, help="Radians per keypress")
    args = parser.parse_args()

    config = TeleopConfig(
        record=args.record,
        output_dir=args.output,
        step_size_rad=args.step_size,
    )
    teleop = TeleopController(config)
    teleop.start()
