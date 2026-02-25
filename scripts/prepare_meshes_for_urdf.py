#!/usr/bin/env python3
"""Prepare meshes for URDF: re-center at link frame origin, rotate Y-up -> Z-up.

The XArm.STEP assembly uses Y-up coordinates. URDF convention uses Z-up.
Each mesh is translated so its link's joint origin is at (0,0,0), then
rotated so the arm extends along Z.

Also creates simple box STL meshes for gripper fingers (since the gripper
STEP files are for a different gripper design).
"""

import os
import trimesh
import numpy as np

OUTPUT_DIR = os.path.expanduser("~/dev/Alice-001/meshes")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")
COLLISION_DIR = os.path.join(OUTPUT_DIR, "collision")

# Rotation matrix: Y-up -> Z-up (rotate -90deg around X)
# x' = x, y' = z, z' = -y  ... wait
# To go from Y-up to Z-up, we rotate +90 degrees around X:
# x' = x, y' = -z, z' = y
ROT_Y_TO_Z = np.array([
    [1,  0, 0],
    [0,  0, -1],
    [0,  1, 0],
], dtype=float)

# Joint positions in assembly coordinates (Y-up, meters)
# Estimated from mesh boundary analysis of XArm.STEP
JOINT_POSITIONS_ASSEMBLY = {
    "base_link":      np.array([0.0, 0.0,   0.0]),    # World origin
    "shoulder_link":  np.array([0.0, 0.065,  0.0]),    # Turntable center
    "upper_arm_link": np.array([0.0, 0.080,  0.0]),    # Shoulder pivot
    "lower_arm_link": np.array([0.0, 0.151,  0.0]),    # Elbow pivot
    # Note: "lower_arm" in our mesh is between elbow and wrist
    "wrist_link":     np.array([0.0, 0.237,  0.0]),    # Wrist pitch pivot
}

# Where each link's mesh should be centered relative to its own frame
# (in assembly coordinates before rotation)
# This is the joint position that defines this link's frame origin
LINK_FRAME_ORIGIN = {
    "base_link":      np.array([0.0, 0.0,   0.0]),
    "shoulder_link":  np.array([0.0, 0.065,  0.0]),
    "upper_arm_link": np.array([0.0, 0.151,  0.0]),
    "lower_arm_link": np.array([0.0, 0.237,  0.0]),
    "wrist_link":     np.array([0.0, 0.280,  0.0]),
}


def create_box_stl(size, filename):
    """Create a simple box STL for gripper fingers."""
    box = trimesh.creation.box(extents=size)
    box.export(filename)
    return box


def process_link(name):
    """Re-center and rotate a link mesh."""
    for subdir in [VISUAL_DIR, COLLISION_DIR]:
        path = os.path.join(subdir, f"{name}.stl")
        if not os.path.exists(path):
            print(f"  {subdir}/{name}.stl not found, skipping")
            continue

        mesh = trimesh.load(path, force="mesh")

        # Translate: move link frame origin to (0,0,0)
        origin = LINK_FRAME_ORIGIN[name]
        mesh.vertices -= origin

        # Rotate: Y-up -> Z-up
        mesh.vertices = mesh.vertices @ ROT_Y_TO_Z.T

        # Save back
        mesh.export(path)

        ext = mesh.bounds[1] - mesh.bounds[0]
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2
        print(f"  {os.path.basename(subdir)}/{name}.stl: "
              f"extent={np.array2string(ext*1000, precision=1)}mm, "
              f"center={np.array2string(center*1000, precision=1)}mm")


def main():
    print("Re-centering and rotating arm link meshes...")
    for name in LINK_FRAME_ORIGIN:
        print(f"\n{name}:")
        process_link(name)

    # Create gripper finger box meshes
    print("\nCreating gripper box primitives...")

    # Gripper base: small box
    size_base = [0.030, 0.050, 0.020]  # 30x50x20mm
    for subdir in [VISUAL_DIR, COLLISION_DIR]:
        create_box_stl(size_base, os.path.join(subdir, "gripper_base.stl"))
    print(f"  gripper_base: {size_base[0]*1000:.0f}x{size_base[1]*1000:.0f}x{size_base[2]*1000:.0f}mm box")

    # Fingers: thin boxes
    size_finger = [0.016, 0.016, 0.040]  # 16x16x40mm
    for subdir in [VISUAL_DIR, COLLISION_DIR]:
        create_box_stl(size_finger, os.path.join(subdir, "left_finger.stl"))
        create_box_stl(size_finger, os.path.join(subdir, "right_finger.stl"))
    print(f"  left/right_finger: {size_finger[0]*1000:.0f}x{size_finger[1]*1000:.0f}x{size_finger[2]*1000:.0f}mm box")

    print("\nAll meshes ready for URDF!")


if __name__ == "__main__":
    main()
