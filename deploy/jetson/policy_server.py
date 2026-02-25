"""TensorRT policy inference server for Jetson deployment.

Runs the vision policy at ~27 Hz (or 2 Hz with action chunking):
  Camera → Preprocess → TensorRT → Action chunks → Servo commands

Supports both state-only and vision policies.

Usage:
    # Vision policy
    server = PolicyServer(
        engine_path="/ssd/alice/models/policy_vision.engine",
        mode="vision",
    )
    server.start()  # Blocks, runs control loop

    # State-only policy (for testing)
    server = PolicyServer(
        engine_path="/ssd/alice/models/policy_state.engine",
        mode="state",
    )
"""

import time
import signal
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .camera import CameraCapture, CameraConfig
from .servo_controller import ServoController, ServoConfig
from .safety import SafetyConfig
from .data_logger import DataLogger, LogConfig


@dataclass
class PolicyConfig:
    """Policy server configuration."""
    engine_path: str = "/ssd/alice/models/policy_vision.engine"
    mode: str = "vision"  # "vision" or "state"

    # Control rates
    policy_hz: float = 27.0  # How often to run inference
    servo_hz: float = 50.0   # How often to command servos

    # Action chunking
    action_chunk_size: int = 1  # Set >1 for action chunking (e.g., 16)
    chunk_dt: float = 0.033    # Time between chunk waypoints

    # Action scaling (matches sim config)
    action_scale: float = 0.5

    # Episode management
    episode_timeout_s: float = 60.0
    auto_reset: bool = True  # Auto-reset after episode ends

    # Logging
    log_dir: str = "/ssd/alice/logs"
    log_episodes: bool = True


class TensorRTEngine:
    """Wrapper for TensorRT engine inference."""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self._engine = None
        self._context = None
        self._bindings = None

    def load(self):
        """Load TensorRT engine from file."""
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
            print(f"[TRT] Loaded engine: {self.engine_path}")

            # Print binding info
            for i in range(self._engine.num_bindings):
                name = self._engine.get_binding_name(i)
                shape = self._engine.get_binding_shape(i)
                is_input = self._engine.binding_is_input(i)
                print(f"  {'Input' if is_input else 'Output'} {name}: {shape}")

        except ImportError:
            print("[TRT] TensorRT not available, using ONNX Runtime fallback")
            self._load_onnx_fallback()

    def _load_onnx_fallback(self):
        """Fallback to ONNX Runtime when TensorRT isn't available."""
        import onnxruntime as ort

        onnx_path = self.engine_path.replace(".engine", ".onnx")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[ORT] Loaded ONNX model: {onnx_path}")

    def infer_vision(self, images: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        """Run vision policy inference.

        Args:
            images: (1, num_frames*3, 224, 224) float32
            proprio: (1, 14) float32

        Returns:
            actions: (1, 7) float32
        """
        if hasattr(self, "_ort_session"):
            return self._infer_ort({"images": images, "proprioception": proprio})
        return self._infer_trt({"images": images, "proprioception": proprio})

    def infer_state(self, observations: np.ndarray) -> np.ndarray:
        """Run state-only policy inference.

        Args:
            observations: (1, 28) float32

        Returns:
            actions: (1, 7) float32
        """
        if hasattr(self, "_ort_session"):
            return self._infer_ort({"observations": observations})
        return self._infer_trt({"observations": observations})

    def _infer_ort(self, inputs: dict) -> np.ndarray:
        """Inference via ONNX Runtime."""
        output_names = [o.name for o in self._ort_session.get_outputs()]
        results = self._ort_session.run(output_names, inputs)
        return results[0]

    def _infer_trt(self, inputs: dict) -> np.ndarray:
        """Inference via TensorRT."""
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        # Allocate device memory and transfer inputs
        d_inputs = {}
        for name, arr in inputs.items():
            d_buf = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(d_buf, arr)
            d_inputs[name] = d_buf

        # Allocate output
        output_shape = (1, 7)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Build bindings list
        bindings = []
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            if name in d_inputs:
                bindings.append(int(d_inputs[name]))
            else:
                bindings.append(int(d_output))

        # Execute
        self._context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)

        # Cleanup
        for d_buf in d_inputs.values():
            d_buf.free()
        d_output.free()

        return output


class PolicyServer:
    """Main control loop: camera → policy → servos."""

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.engine = TensorRTEngine(self.config.engine_path)
        self.servo = ServoController()
        self.camera = None
        self.logger = None

        if self.config.mode == "vision":
            self.camera = CameraCapture()

        if self.config.log_episodes:
            self.logger = DataLogger(LogConfig(log_dir=self.config.log_dir))

        self._running = False
        self._episode_count = 0
        self._episode_step = 0

    def start(self):
        """Start the control loop (blocking)."""
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"\n{'='*60}")
        print(f"Alice Policy Server")
        print(f"  Mode: {self.config.mode}")
        print(f"  Engine: {self.config.engine_path}")
        print(f"  Policy rate: {self.config.policy_hz} Hz")
        print(f"  Servo rate: {self.config.servo_hz} Hz")
        print(f"{'='*60}\n")

        # Initialize components
        self.engine.load()
        self.servo.connect()

        if self.camera:
            self.camera.start()
            time.sleep(1.0)  # Let camera warm up

        self.servo.go_home(duration_ms=2000)
        time.sleep(2.0)

        self._running = True
        self._run_control_loop()

    def stop(self):
        """Stop the control loop."""
        self._running = False
        self.servo.hold_position()
        if self.camera:
            self.camera.stop()
        self.servo.disconnect()
        if self.logger:
            self.logger.close()
        print("[Server] Stopped")

    def _run_control_loop(self):
        """Main control loop."""
        policy_dt = 1.0 / self.config.policy_hz
        self._episode_start = time.time()
        self._episode_step = 0

        while self._running:
            loop_start = time.time()

            # Safety check
            stop, reason = self.servo.safety.should_stop()
            if stop:
                print(f"[Server] SAFETY STOP: {reason}")
                self.servo.hold_position()
                self._running = False
                break

            # Get observations and run inference
            if self.config.mode == "vision":
                actions = self._step_vision()
            else:
                actions = self._step_state()

            # Apply actions
            if actions is not None:
                self._apply_actions(actions)

            self._episode_step += 1

            # Episode timeout check
            episode_elapsed = time.time() - self._episode_start
            if episode_elapsed > self.config.episode_timeout_s:
                print(f"[Server] Episode {self._episode_count} timeout ({episode_elapsed:.1f}s)")
                self._end_episode(success=False)

            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step_vision(self) -> Optional[np.ndarray]:
        """Run one vision policy step."""
        # Get camera frames
        images = self.camera.get_stacked_frames()  # (num_frames, 3, H, W)
        images = images.reshape(1, -1, 224, 224).astype(np.float32)

        # Get proprioception
        joint_pos = self.servo.read_positions_normalized()
        # Approximate velocity from position delta
        joint_vel = [0.0] * 7  # TODO: compute from position history
        proprio = np.array([joint_pos + joint_vel], dtype=np.float32)

        # Inference
        actions = self.engine.infer_vision(images, proprio)

        # Log
        if self.logger:
            self.logger.log_step(
                images=images,
                joint_pos=np.array(joint_pos),
                actions=actions[0],
            )

        return actions[0]

    def _step_state(self) -> Optional[np.ndarray]:
        """Run one state-only policy step (for testing without camera)."""
        joint_pos = self.servo.read_positions_normalized()
        joint_vel = [0.0] * 7
        # For state-only, we'd need ball position from external tracking
        # For testing, use zeros for task-specific observations
        ball_obs = [0.0] * 14  # ee_pos, ball_pos, ee_to_ball, ball_to_target, gripper, contact
        obs = np.array([joint_pos + joint_vel + ball_obs], dtype=np.float32)
        actions = self.engine.infer_state(obs)
        return actions[0]

    def _apply_actions(self, actions: np.ndarray):
        """Convert policy actions to servo commands.

        Args:
            actions: (7,) action deltas from policy
        """
        # Scale actions (matches sim: delta targets)
        current = self.servo.read_positions_radians()
        dt = 1.0 / self.config.policy_hz
        speed_scales = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1]  # Gripper slower

        targets = []
        for i in range(7):
            delta = speed_scales[i] * dt * float(actions[i]) * self.config.action_scale
            targets.append(current[i] + delta)

        self.servo.move_to_radians(targets)

    def _end_episode(self, success: bool):
        """Handle episode end: log, reset, start new episode."""
        self.servo.safety.record_episode(success)
        self._episode_count += 1
        print(f"[Server] Episode {self._episode_count}: {'SUCCESS' if success else 'TIMEOUT'}")

        if self.config.auto_reset:
            self.servo.go_home(duration_ms=1000)
            time.sleep(1.5)
            self._episode_start = time.time()
            self._episode_step = 0

            if self.logger:
                self.logger.new_episode()

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n[Server] Shutdown signal received")
        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alice Policy Server")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine or .onnx")
    parser.add_argument("--mode", choices=["vision", "state"], default="vision")
    parser.add_argument("--policy-hz", type=float, default=27.0)
    args = parser.parse_args()

    config = PolicyConfig(
        engine_path=args.engine,
        mode=args.mode,
        policy_hz=args.policy_hz,
    )
    server = PolicyServer(config)
    server.start()
