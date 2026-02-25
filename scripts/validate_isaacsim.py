"""Validate Alice 001 USD in Isaac Sim (no Isaac Lab dependency)."""
import sys
import os
import time

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

log("VALIDATE: Starting...")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdPhysics, UsdGeom, Usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

log("\n=== Alice 001 USD Validation ===\n")

# Create world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Load USD
usd_path = "/workspace/Alice-001/usd/alice_001.usd"
log("Loading {}...".format(usd_path))
prim = add_reference_to_stage(usd_path=usd_path, prim_path="/World/Alice001")
log("Prim loaded: {}".format(prim.GetPath()))

# Inspect the stage
stage = omni.usd.get_context().get_stage()
log("\nPrim hierarchy:")
for p in stage.Traverse():
    path = str(p.GetPath())
    if "/World/Alice001" in path:
        apis = []
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("ArticulationRoot")
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if p.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        if p.IsA(UsdPhysics.RevoluteJoint):
            apis.append("RevoluteJoint")
        if p.IsA(UsdGeom.Mesh):
            apis.append("Mesh")
        api_str = ""
        if apis:
            api_str = " [" + ", ".join(apis) + "]"
        # Only print important prims
        if apis or path.count("/") <= 5:
            log("  {}{}".format(path, api_str))

# Reset and simulate
log("\nResetting simulation...")
world.reset()

log("Running gravity drop test (500 steps)...")
start = time.time()
for i in range(500):
    world.step(render=False)
    if i % 100 == 0:
        log("  Step {}/500".format(i))

elapsed = time.time() - start
log("\n500 steps in {:.2f}s ({:.0f} steps/sec)".format(elapsed, 500 / elapsed))
log("PASS: Simulation ran 500 steps without crash")

world.stop()
simulation_app.close()
log("\n=== Validation Complete ===")
