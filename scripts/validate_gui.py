"""Alice 001 USD validation — GUI mode, native Isaac Lab.

Run with:
  cd ~/IsaacLab && ./isaaclab.sh -p ~/Alice-001/scripts/validate_gui.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import time
import omni.usd
from pxr import UsdPhysics, UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
import numpy as np

print("\n=== Alice 001 — Interactive Validation ===\n")

# Create world with 60Hz physics
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
world.scene.add_default_ground_plane()

# Load Alice 001 USD
usd_path = "/home/ubuntu/Alice-001/usd/alice_001.usd"
print("Loading {}...".format(usd_path))
add_reference_to_stage(usd_path=usd_path, prim_path="/World/Alice001")

# Register as articulation so we can control joints
robot = world.scene.add(
    Articulation(prim_path="/World/Alice001", name="alice_001")
)

print("Resetting world...")
world.reset()

# Print joint info
num_joints = robot.num_dof
dof_names = robot.dof_names
print("\nRobot DOFs ({}):".format(num_joints))
for i, name in enumerate(dof_names):
    pos = robot.get_joint_positions()[i]
    print("  [{}] {} = {:.3f} rad".format(i, name, pos))

print("\nRunning simulation — robot visible in viewport...")
print("Press Ctrl+C to stop\n")

step = 0
start = time.time()

try:
    while simulation_app.is_running():
        world.step(render=True)

        # Every 3 seconds, sweep joints to show the robot is alive
        if step % 180 == 0:
            t = step / 60.0
            # Gentle sinusoidal sweep on arm joints
            targets = np.zeros(num_joints)
            for i, name in enumerate(dof_names):
                if "base" in name:
                    targets[i] = 0.3 * np.sin(0.5 * t)
                elif "shoulder" in name:
                    targets[i] = 0.2 * np.sin(0.3 * t)
                elif "elbow" in name:
                    targets[i] = -0.4 + 0.2 * np.sin(0.4 * t)
                elif "wrist" in name:
                    targets[i] = 0.15 * np.sin(0.6 * t)
                elif "finger" in name:
                    targets[i] = 0.1 * abs(np.sin(0.5 * t))

            robot.set_joint_position_targets(targets)

            elapsed = time.time() - start
            fps = step / elapsed if elapsed > 0 else 0
            pos = robot.get_joint_positions()
            print("Step {:5d} | {:.1f}s | {:.0f} steps/s | base={:.2f}rad shoulder={:.2f}rad".format(
                step, elapsed, fps, pos[0] if len(pos) > 0 else 0, pos[1] if len(pos) > 1 else 0
            ))

        step += 1

except KeyboardInterrupt:
    pass

world.stop()
simulation_app.close()
print("\n=== Done ===")
