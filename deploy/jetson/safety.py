"""Safety system for unattended xArm operation on Jetson.

Enforces joint limits, velocity caps, workspace bounds, and watchdog.
All safety checks run on CPU at >100 Hz — never on the GPU path.

Usage:
    safety = SafetyController()
    safe_targets = safety.check(joint_targets, current_pos)
    if safety.should_stop():
        arm.hold_position()
"""

import math
import time
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JointLimits:
    """Joint limits matching URDF for xArm 1S."""
    # (lower_rad, upper_rad) for each joint
    limits: list = field(default_factory=lambda: [
        (-math.pi, math.pi),        # base_joint
        (-math.pi / 2, math.pi / 2),  # shoulder_joint
        (-2.094, 0.0),               # elbow_joint (-120° to 0°)
        (-math.pi / 2, math.pi / 2),  # wrist_pitch_joint
        (-math.pi, math.pi),        # wrist_roll_joint
        (0.0, 0.524),               # left_finger_joint (0 to 30°)
        (-0.524, 0.0),              # right_finger_joint (-30° to 0°)
    ])

    # Software limits tighter than URDF for safety margin
    margin_rad: float = 0.05  # ~3 degrees margin from hardware limit


@dataclass
class SafetyConfig:
    """Safety parameters for unattended operation."""
    # Velocity limits
    max_joint_velocity_rad_s: float = 4.19  # 80% of URDF max (5.24 rad/s)

    # Workspace bounds (meters, relative to base)
    workspace_radius: float = 0.30  # 30cm sphere
    workspace_z_min: float = 0.0    # Don't go below table
    workspace_z_max: float = 0.40   # Don't reach too high

    # Watchdog
    heartbeat_timeout_s: float = 0.5  # Hold position if no command for 500ms

    # Episode limits
    episode_timeout_s: float = 60.0
    min_success_rate: float = 0.10  # Stop if <10% for consecutive window
    success_rate_window: int = 10   # Number of episodes to track

    # Thermal (Jetson-specific)
    thermal_warn_c: float = 75.0
    thermal_stop_c: float = 85.0


class SafetyController:
    """Enforces all safety constraints for the xArm."""

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.joint_limits = JointLimits()

        # Watchdog
        self._last_heartbeat = time.time()
        self._watchdog_triggered = False

        # Episode tracking
        self._episode_results = []  # List of bools (success/failure)
        self._estop = False

        # Thermal monitoring
        self._temperature = 0.0
        self._thermal_thread = None
        self._monitoring = False

    def start_monitoring(self):
        """Start background thermal monitoring."""
        self._monitoring = True
        self._thermal_thread = threading.Thread(target=self._monitor_thermal, daemon=True)
        self._thermal_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False

    def heartbeat(self):
        """Call this every control cycle to prevent watchdog trigger."""
        self._last_heartbeat = time.time()
        self._watchdog_triggered = False

    def check_joints(self, targets: list[float], current: list[float], dt: float) -> list[float]:
        """Clamp joint targets to safe limits and velocity constraints.

        Args:
            targets: Desired joint positions (7 values, radians)
            current: Current joint positions (7 values, radians)
            dt: Time since last command (seconds)

        Returns:
            Safe joint positions (clamped)
        """
        safe = []
        max_delta = self.config.max_joint_velocity_rad_s * dt

        for i, (target, cur) in enumerate(zip(targets, current)):
            lo, hi = self.joint_limits.limits[i]
            margin = self.joint_limits.margin_rad

            # Clamp to joint limits with margin
            clamped = max(lo + margin, min(hi - margin, target))

            # Velocity limit: max delta per timestep
            delta = clamped - cur
            if abs(delta) > max_delta:
                clamped = cur + max_delta * (1.0 if delta > 0 else -1.0)

            safe.append(clamped)

        return safe

    def check_workspace(self, ee_pos: tuple[float, float, float]) -> bool:
        """Check if end-effector is within workspace bounds.

        Args:
            ee_pos: (x, y, z) end-effector position relative to base (meters)

        Returns:
            True if within bounds
        """
        x, y, z = ee_pos
        radius = math.sqrt(x * x + y * y + z * z)

        if radius > self.config.workspace_radius:
            return False
        if z < self.config.workspace_z_min or z > self.config.workspace_z_max:
            return False
        return True

    def should_stop(self) -> tuple[bool, str]:
        """Check all stop conditions.

        Returns:
            (should_stop, reason) tuple
        """
        if self._estop:
            return True, "E-STOP activated"

        # Watchdog
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self.config.heartbeat_timeout_s:
            self._watchdog_triggered = True
            return True, f"Watchdog: no heartbeat for {elapsed:.1f}s"

        # Thermal
        if self._temperature > self.config.thermal_stop_c:
            return True, f"Thermal: {self._temperature:.1f}°C exceeds {self.config.thermal_stop_c}°C"

        # Success rate
        if len(self._episode_results) >= self.config.success_rate_window:
            recent = self._episode_results[-self.config.success_rate_window:]
            rate = sum(recent) / len(recent)
            if rate < self.config.min_success_rate:
                return True, f"Success rate {rate:.0%} < {self.config.min_success_rate:.0%} over {self.config.success_rate_window} episodes"

        return False, ""

    def record_episode(self, success: bool):
        """Record episode outcome for success rate tracking."""
        self._episode_results.append(success)

    def estop(self):
        """Emergency stop — requires manual reset."""
        self._estop = True

    def reset_estop(self):
        """Clear e-stop after human verification."""
        self._estop = False
        self._watchdog_triggered = False

    def get_temperature(self) -> float:
        """Get current Jetson temperature."""
        return self._temperature

    def _monitor_thermal(self):
        """Background thread: poll Jetson thermal zones."""
        while self._monitoring:
            try:
                result = subprocess.run(
                    ["cat", "/sys/class/thermal/thermal_zone0/temp"],
                    capture_output=True, text=True, timeout=1,
                )
                if result.returncode == 0:
                    self._temperature = int(result.stdout.strip()) / 1000.0
                    if self._temperature > self.config.thermal_warn_c:
                        print(f"[SAFETY] Thermal warning: {self._temperature:.1f}°C")
            except Exception:
                pass
            time.sleep(5)

    @property
    def status(self) -> dict:
        """Current safety status summary."""
        stop, reason = self.should_stop()
        return {
            "estop": self._estop,
            "watchdog_triggered": self._watchdog_triggered,
            "temperature_c": self._temperature,
            "episodes_recorded": len(self._episode_results),
            "recent_success_rate": (
                sum(self._episode_results[-self.config.success_rate_window:])
                / max(1, len(self._episode_results[-self.config.success_rate_window:]))
                if self._episode_results else 0.0
            ),
            "should_stop": stop,
            "stop_reason": reason,
        }
