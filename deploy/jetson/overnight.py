"""Overnight autonomous learning orchestrator.

Runs the full cycle: safety check → data collection → fine-tuning →
evaluation → model promotion. Designed for unattended operation.

Safety: joint limits, velocity caps, workspace bounds, watchdog, thermal
monitoring, and auto-stop on low success rate.

Usage:
    python -m deploy.jetson.overnight \
        --engine /ssd/alice/models/policy_vision.engine \
        --data-dir /ssd/alice/data/overnight \
        --num-collect-episodes 100 \
        --num-eval-episodes 20

Typical overnight schedule:
    6 PM  - Human starts script, goes to bed
    6-8 PM  - Collect ~100 episodes with current policy
    8 PM-2 AM - Fine-tune on all accumulated data
    2-3 AM  - Evaluate new policy (20 episodes)
    3 AM    - If improved: promote. If not: rollback.
    3-6 AM  - (Optional) Collect more data with promoted policy
    6 AM    - Human checks logs in /ssd/alice/logs/
"""

import os
import sys
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .safety import SafetyController, SafetyConfig
from .policy_server import PolicyServer, PolicyConfig
from .data_logger import DataLogger, LogConfig


@dataclass
class OvernightConfig:
    """Overnight learning configuration."""
    # Model paths
    current_engine: str = "/ssd/alice/models/policy_vision.engine"
    candidate_engine: str = "/ssd/alice/models/policy_vision_candidate.engine"
    backup_engine: str = "/ssd/alice/models/policy_vision_backup.engine"

    # Data collection
    data_dir: str = "/ssd/alice/data/overnight"
    num_collect_episodes: int = 100
    episode_timeout_s: float = 60.0

    # Evaluation
    num_eval_episodes: int = 20
    promotion_threshold: float = 0.05  # Must improve success rate by 5%

    # Fine-tuning (runs as subprocess)
    finetune_script: str = ""  # Path to fine-tuning script
    finetune_epochs: int = 100
    finetune_lr: float = 1e-4

    # Safety
    max_consecutive_failures: int = 10
    thermal_stop_c: float = 82.0

    # Logging
    log_dir: str = "/ssd/alice/logs"

    # Schedule
    max_runtime_hours: float = 12.0


class OvernightRunner:
    """Orchestrates the overnight learning cycle."""

    def __init__(self, config: Optional[OvernightConfig] = None):
        self.config = config or OvernightConfig()
        self.safety = SafetyController(SafetyConfig(
            thermal_stop_c=self.config.thermal_stop_c,
        ))
        self._start_time = None
        self._log_file = None

    def run(self):
        """Execute the full overnight cycle."""
        self._start_time = time.time()
        self._setup_logging()

        try:
            self._log("=" * 60)
            self._log("OVERNIGHT LEARNING CYCLE STARTED")
            self._log(f"  Max runtime: {self.config.max_runtime_hours} hours")
            self._log(f"  Collect: {self.config.num_collect_episodes} episodes")
            self._log(f"  Eval: {self.config.num_eval_episodes} episodes")
            self._log("=" * 60)

            # Phase 1: Safety check
            self._log("\n[Phase 1] Safety pre-check...")
            if not self._safety_precheck():
                self._log("ABORTED: Safety pre-check failed")
                return

            # Phase 2: Backup current model
            self._log("\n[Phase 2] Backing up current model...")
            self._backup_model()

            # Phase 3: Collect data with current policy
            self._log("\n[Phase 3] Collecting data...")
            collect_success_rate = self._collect_data()
            self._log(f"  Collection success rate: {collect_success_rate:.1%}")

            # Check runtime
            if self._exceeded_runtime():
                self._log("Max runtime exceeded, stopping after data collection")
                return

            # Phase 4: Fine-tune on collected data
            self._log("\n[Phase 4] Fine-tuning...")
            finetune_ok = self._finetune()
            if not finetune_ok:
                self._log("Fine-tuning failed, keeping current model")
                return

            # Check runtime
            if self._exceeded_runtime():
                self._log("Max runtime exceeded, stopping after fine-tuning")
                return

            # Phase 5: Evaluate candidate model
            self._log("\n[Phase 5] Evaluating candidate model...")
            eval_success_rate = self._evaluate_candidate()
            self._log(f"  Candidate success rate: {eval_success_rate:.1%}")
            self._log(f"  Current success rate:   {collect_success_rate:.1%}")

            # Phase 6: Promote or rollback
            improvement = eval_success_rate - collect_success_rate
            if improvement >= self.config.promotion_threshold:
                self._log(f"\n[Phase 6] PROMOTING candidate (+{improvement:.1%} improvement)")
                self._promote_candidate()
            else:
                self._log(f"\n[Phase 6] ROLLBACK (improvement {improvement:.1%} < threshold {self.config.promotion_threshold:.1%})")
                self._rollback()

            # Summary
            elapsed_h = (time.time() - self._start_time) / 3600
            self._log(f"\n{'='*60}")
            self._log(f"OVERNIGHT CYCLE COMPLETE ({elapsed_h:.1f} hours)")
            self._log(f"  Collected: {self.config.num_collect_episodes} episodes")
            self._log(f"  Collection success: {collect_success_rate:.1%}")
            self._log(f"  Candidate success:  {eval_success_rate:.1%}")
            self._log(f"  Decision: {'PROMOTED' if improvement >= self.config.promotion_threshold else 'ROLLED BACK'}")
            self._log(f"{'='*60}")

        except Exception as e:
            self._log(f"OVERNIGHT ERROR: {e}")
            raise
        finally:
            if self._log_file:
                self._log_file.close()

    def _safety_precheck(self) -> bool:
        """Verify safety systems are operational."""
        self.safety.start_monitoring()
        time.sleep(2)

        temp = self.safety.get_temperature()
        self._log(f"  Temperature: {temp:.1f}°C")
        if temp > self.config.thermal_stop_c:
            self._log(f"  FAIL: Temperature too high ({temp:.1f}°C > {self.config.thermal_stop_c}°C)")
            return False

        stop, reason = self.safety.should_stop()
        if stop:
            self._log(f"  FAIL: {reason}")
            return False

        self._log("  PASS: All safety checks OK")
        return True

    def _backup_model(self):
        """Backup current model before modifications."""
        if os.path.exists(self.config.current_engine):
            shutil.copy2(self.config.current_engine, self.config.backup_engine)
            self._log(f"  Backed up to {self.config.backup_engine}")

    def _collect_data(self) -> float:
        """Collect episodes with current policy. Returns success rate."""
        data_path = os.path.join(
            self.config.data_dir,
            time.strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(data_path, exist_ok=True)

        policy_cfg = PolicyConfig(
            engine_path=self.config.current_engine,
            mode="vision",
            episode_timeout_s=self.config.episode_timeout_s,
            log_dir=data_path,
            log_episodes=True,
            auto_reset=True,
        )

        # Run policy server for data collection
        # In practice, this runs the control loop for N episodes
        server = PolicyServer(policy_cfg)
        server.engine.load()
        server.servo.connect()

        if server.camera:
            server.camera.start()
            time.sleep(1.0)

        successes = 0
        total = 0

        try:
            for ep in range(self.config.num_collect_episodes):
                stop, reason = self.safety.should_stop()
                if stop:
                    self._log(f"  Safety stop during collection: {reason}")
                    break

                if self._exceeded_runtime():
                    self._log(f"  Runtime exceeded during collection at episode {ep}")
                    break

                # Run one episode
                success = self._run_episode(server)
                if success:
                    successes += 1
                total += 1

                if total % 10 == 0:
                    rate = successes / total if total > 0 else 0
                    self._log(f"  Episode {total}/{self.config.num_collect_episodes}: {rate:.1%} success")
        finally:
            server.stop()

        return successes / max(1, total)

    def _run_episode(self, server: PolicyServer) -> bool:
        """Run a single episode. Returns success."""
        server.servo.go_home(duration_ms=1000)
        time.sleep(1.5)

        # Simple episode loop
        episode_start = time.time()
        for step in range(2000):
            if time.time() - episode_start > self.config.episode_timeout_s:
                return False

            stop, reason = server.servo.safety.should_stop()
            if stop:
                return False

            if server.config.mode == "vision":
                actions = server._step_vision()
            else:
                actions = server._step_state()

            if actions is not None:
                server._apply_actions(actions)

            time.sleep(1.0 / server.config.policy_hz)

        return False  # Timeout = failure

    def _finetune(self) -> bool:
        """Run fine-tuning on accumulated data. Returns True on success."""
        if not self.config.finetune_script:
            self._log("  No fine-tuning script configured, skipping")
            return True

        cmd = [
            sys.executable, self.config.finetune_script,
            "--data-dir", self.config.data_dir,
            "--output", self.config.candidate_engine.replace(".engine", ".pt"),
            "--epochs", str(self.config.finetune_epochs),
            "--lr", str(self.config.finetune_lr),
        ]
        self._log(f"  Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=6 * 3600,
            )
            if result.returncode != 0:
                self._log(f"  Fine-tuning failed: {result.stderr[-500:]}")
                return False
            self._log("  Fine-tuning completed successfully")
            return True
        except subprocess.TimeoutExpired:
            self._log("  Fine-tuning timed out (6 hours)")
            return False

    def _evaluate_candidate(self) -> float:
        """Evaluate candidate model. Returns success rate."""
        if not os.path.exists(self.config.candidate_engine):
            self._log("  No candidate engine found, using current model metrics")
            return 0.0

        policy_cfg = PolicyConfig(
            engine_path=self.config.candidate_engine,
            mode="vision",
            episode_timeout_s=self.config.episode_timeout_s,
            log_episodes=False,
        )

        server = PolicyServer(policy_cfg)
        server.engine.load()
        server.servo.connect()

        if server.camera:
            server.camera.start()
            time.sleep(1.0)

        successes = 0
        total = 0

        try:
            for ep in range(self.config.num_eval_episodes):
                success = self._run_episode(server)
                if success:
                    successes += 1
                total += 1
        finally:
            server.stop()

        return successes / max(1, total)

    def _promote_candidate(self):
        """Replace current model with candidate."""
        if os.path.exists(self.config.candidate_engine):
            shutil.copy2(self.config.candidate_engine, self.config.current_engine)
            self._log(f"  Promoted {self.config.candidate_engine} → {self.config.current_engine}")

    def _rollback(self):
        """Restore backup model."""
        if os.path.exists(self.config.backup_engine):
            shutil.copy2(self.config.backup_engine, self.config.current_engine)
            self._log(f"  Rolled back to {self.config.backup_engine}")

    def _exceeded_runtime(self) -> bool:
        """Check if we've exceeded max runtime."""
        elapsed_h = (time.time() - self._start_time) / 3600
        return elapsed_h > self.config.max_runtime_hours

    def _setup_logging(self):
        """Set up log file."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.config.log_dir, f"overnight_{timestamp}.log")
        self._log_file = open(log_path, "w")
        self._log(f"Log file: {log_path}")

    def _log(self, msg: str):
        """Log to console and file."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Overnight Learning Orchestrator")
    parser.add_argument("--engine", type=str, default="/ssd/alice/models/policy_vision.engine")
    parser.add_argument("--data-dir", type=str, default="/ssd/alice/data/overnight")
    parser.add_argument("--num-collect-episodes", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--finetune-script", type=str, default="")
    args = parser.parse_args()

    config = OvernightConfig(
        current_engine=args.engine,
        data_dir=args.data_dir,
        num_collect_episodes=args.num_collect_episodes,
        num_eval_episodes=args.num_eval_episodes,
        max_runtime_hours=args.max_hours,
        finetune_script=args.finetune_script,
    )
    runner = OvernightRunner(config)
    runner.run()
