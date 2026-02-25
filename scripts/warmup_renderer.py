"""Warm up Isaac Sim rendering path to populate first-run caches."""

import argparse
import time

parser = argparse.ArgumentParser(description="Warm up renderer")
parser.add_argument("--seconds", type=int, default=90)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# Keep app alive briefly so renderer/extension caches initialize.
end_t = time.time() + max(1, args_cli.seconds)
while simulation_app.is_running() and time.time() < end_t:
    simulation_app.update()

simulation_app.close()
print("Renderer warmup done")
