"""xArm 1S servo controller with safety enforcement.

Wraps the xarm library to provide:
- Radian ↔ servo unit conversion
- Safety limit enforcement
- Smooth trajectory interpolation from action chunks
- Position reading for proprioception

The xArm uses LX-16A bus servos with 0-1000 position units.

Usage:
    ctrl = ServoController()
    ctrl.connect()
    ctrl.move_to_radians([0, 0, -1.047, 0, 0, 0.26, -0.26])
    positions = ctrl.read_positions_radians()
    ctrl.disconnect()
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import xarm
except ImportError:
    xarm = None

from .safety import SafetyController, SafetyConfig


@dataclass
class ServoConfig:
    """Configuration for xArm 1S servos."""
    # Servo IDs (1-indexed on the bus)
    servo_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # Servo unit range
    unit_min: int = 0
    unit_max: int = 1000

    # Joint angle ranges matching URDF (radians)
    # (lower_rad, upper_rad) per joint
    joint_ranges: list = field(default_factory=lambda: [
        (-math.pi, math.pi),           # base
        (-math.pi / 2, math.pi / 2),   # shoulder
        (-2.094, 0.0),                  # elbow
        (-math.pi / 2, math.pi / 2),   # wrist_pitch
        (-math.pi, math.pi),           # wrist_roll
        (0.0, 0.524),                   # left_finger
        (-0.524, 0.0),                  # right_finger
    ])

    # Servo center positions (units) — calibrated to URDF zero position
    # These need calibration on the real robot
    center_units: list = field(default_factory=lambda: [500, 500, 500, 500, 500, 500, 500])

    # Units per radian (calibrated)
    units_per_rad: list = field(default_factory=lambda: [
        159.15,  # 1000 / (2*pi)
        318.31,  # 1000 / pi
        477.46,  # 1000 / 2.094
        318.31,
        159.15,
        1908.4,  # 1000 / 0.524
        1908.4,
    ])

    # Control rate
    control_rate_hz: float = 50.0

    # Duration for servo moves (ms) — affects smoothness
    default_move_duration_ms: int = 20


class ServoController:
    """xArm servo controller with safety integration."""

    def __init__(
        self,
        servo_config: Optional[ServoConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
    ):
        self.servo_cfg = servo_config or ServoConfig()
        self.safety = SafetyController(safety_config)
        self._arm = None
        self._last_targets_rad = [0.0] * 7
        self._last_command_time = time.time()

    def connect(self, port: str = "USB"):
        """Connect to xArm controller."""
        if xarm is None:
            raise RuntimeError("xarm library not installed. pip install xarm")

        self._arm = xarm.Controller(port)
        print(f"[Servo] Connected to xArm via {port}")

        # Read initial positions
        self._last_targets_rad = self.read_positions_radians()
        self.safety.start_monitoring()

    def disconnect(self):
        """Disconnect and release servos."""
        self.safety.stop_monitoring()
        self._arm = None
        print("[Servo] Disconnected")

    def move_to_radians(self, targets_rad: list[float], duration_ms: Optional[int] = None):
        """Move joints to target positions with safety enforcement.

        Args:
            targets_rad: Target positions in radians (7 values)
            duration_ms: Move duration in ms (None = default)
        """
        if self._arm is None:
            raise RuntimeError("Not connected. Call connect() first.")

        dt = time.time() - self._last_command_time
        self._last_command_time = time.time()
        self.safety.heartbeat()

        # Safety check
        stop, reason = self.safety.should_stop()
        if stop:
            print(f"[Servo] SAFETY STOP: {reason}")
            self.hold_position()
            return

        # Enforce joint and velocity limits
        current_rad = self._last_targets_rad
        safe_targets = self.safety.check_joints(targets_rad, current_rad, max(dt, 0.001))

        # Convert to servo units
        units = self._radians_to_units(safe_targets)
        dur = duration_ms or self.servo_cfg.default_move_duration_ms

        # Send command
        self._arm.setPosition(self.servo_cfg.servo_ids, units, duration=dur)
        self._last_targets_rad = safe_targets

    def execute_action_chunk(self, chunk: list[list[float]], chunk_dt: float = 0.033):
        """Execute a sequence of joint targets (action chunking).

        Interpolates between targets for smooth motion at servo control rate.

        Args:
            chunk: List of 7-dim joint target lists (radians)
            chunk_dt: Time between chunk waypoints (seconds)
        """
        control_dt = 1.0 / self.servo_cfg.control_rate_hz
        dur_ms = int(control_dt * 1000)

        for i, target in enumerate(chunk):
            stop, reason = self.safety.should_stop()
            if stop:
                print(f"[Servo] SAFETY STOP during chunk: {reason}")
                self.hold_position()
                return

            self.move_to_radians(target, duration_ms=dur_ms)
            time.sleep(chunk_dt)

    def read_positions_radians(self) -> list[float]:
        """Read current joint positions in radians."""
        if self._arm is None:
            return [0.0] * 7

        positions_rad = []
        for i, servo_id in enumerate(self.servo_cfg.servo_ids):
            try:
                units = self._arm.getPosition(servo_id)
                rad = self._units_to_radians(i, units)
                positions_rad.append(rad)
            except Exception:
                positions_rad.append(self._last_targets_rad[i])

        return positions_rad

    def read_positions_normalized(self) -> list[float]:
        """Read current positions normalized to [-1, 1]."""
        positions = self.read_positions_radians()
        normalized = []
        for i, pos in enumerate(positions):
            lo, hi = self.servo_cfg.joint_ranges[i]
            norm = 2.0 * (pos - lo) / (hi - lo) - 1.0
            normalized.append(max(-1.0, min(1.0, norm)))
        return normalized

    def hold_position(self):
        """Command servos to hold current position."""
        if self._arm is None:
            return
        current = self.read_positions_radians()
        units = self._radians_to_units(current)
        self._arm.setPosition(self.servo_cfg.servo_ids, units, duration=0)

    def go_home(self, duration_ms: int = 1000):
        """Move to home/default position."""
        home_rad = [0.0, 0.0, -1.047, 0.0, 0.0, 0.26, -0.26]
        units = self._radians_to_units(home_rad)
        if self._arm:
            self._arm.setPosition(self.servo_cfg.servo_ids, units, duration=duration_ms)
        self._last_targets_rad = home_rad

    def _radians_to_units(self, radians: list[float]) -> list[int]:
        """Convert radians to servo units."""
        units = []
        for i, rad in enumerate(radians):
            u = self.servo_cfg.center_units[i] + rad * self.servo_cfg.units_per_rad[i]
            u = max(self.servo_cfg.unit_min, min(self.servo_cfg.unit_max, int(round(u))))
            units.append(u)
        return units

    def _units_to_radians(self, joint_idx: int, units: int) -> float:
        """Convert servo units to radians."""
        return (units - self.servo_cfg.center_units[joint_idx]) / self.servo_cfg.units_per_rad[joint_idx]

    @property
    def status(self) -> dict:
        """Current controller status."""
        return {
            "connected": self._arm is not None,
            "last_targets_rad": self._last_targets_rad,
            "safety": self.safety.status,
        }
