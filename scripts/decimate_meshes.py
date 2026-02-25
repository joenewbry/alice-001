#!/usr/bin/env python3
"""Decimate Hiwonder xArm 1S meshes for simulation.

Loads STLs from ~/Documents/Hiwonder/, scales from mm to meters,
decimates visual meshes to ~10k-30k faces, computes convex hulls for collision.
"""

import os
import trimesh
import fast_simplification
import numpy as np

HIWONDER_DIR = os.path.expanduser("~/Documents/Hiwonder")
OUTPUT_DIR = os.path.expanduser("~/dev/Alice-001/meshes")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")
COLLISION_DIR = os.path.join(OUTPUT_DIR, "collision")

# Arm links (not gripper — that's handled separately)
ARM_LINKS = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
]

VISUAL_TARGET_FACES = 15000  # Target face count for visual meshes
MM_TO_M = 0.001


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate mesh to target face count using fast_simplification.

    Iteratively decimates if a single pass doesn't reach the target.
    """
    result = mesh.copy()
    for attempt in range(5):
        n_faces = result.faces.shape[0]
        if n_faces <= target_faces * 1.2:
            break
        target_reduction = 1.0 - (target_faces / n_faces)
        # Clamp to valid range
        target_reduction = min(max(target_reduction, 0.01), 0.99)
        verts_out, faces_out = fast_simplification.simplify(
            result.vertices.view(np.ndarray).astype(np.float64),
            result.faces.view(np.ndarray).astype(np.int64),
            target_reduction=target_reduction,
            agg=7,  # More aggressive simplification
        )
        result = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
    return result


def process_mesh(name: str, mesh: trimesh.Trimesh) -> None:
    """Process a single mesh: scale, decimate, compute convex hull."""
    # Scale from mm to meters
    mesh.apply_scale(MM_TO_M)

    bbox = mesh.bounds
    extent = bbox[1] - bbox[0]
    print(f"  {name}: {mesh.faces.shape[0]} faces, extent {extent} m")

    # Visual: decimate
    visual = decimate(mesh, VISUAL_TARGET_FACES)
    print(f"  Visual: {mesh.faces.shape[0]} -> {visual.faces.shape[0]} faces")

    visual_path = os.path.join(VISUAL_DIR, f"{name}.stl")
    visual.export(visual_path)

    # Collision: convex hull
    collision = mesh.convex_hull
    print(f"  Collision: {collision.faces.shape[0]} faces (convex hull)")

    collision_path = os.path.join(COLLISION_DIR, f"{name}.stl")
    collision.export(collision_path)


def main():
    os.makedirs(VISUAL_DIR, exist_ok=True)
    os.makedirs(COLLISION_DIR, exist_ok=True)

    for name in ARM_LINKS:
        stl_path = os.path.join(HIWONDER_DIR, f"{name}.stl")
        if not os.path.exists(stl_path):
            print(f"WARNING: {stl_path} not found, skipping")
            continue

        print(f"\nProcessing {name}...")
        mesh = trimesh.load(stl_path, force="mesh")
        process_mesh(name, mesh)

    # Also process gripper.stl as a whole (for reference)
    gripper_path = os.path.join(HIWONDER_DIR, "gripper.stl")
    if os.path.exists(gripper_path):
        print(f"\nProcessing gripper (whole, for reference)...")
        mesh = trimesh.load(gripper_path, force="mesh")
        mesh.apply_scale(MM_TO_M)
        bbox = mesh.bounds
        extent = bbox[1] - bbox[0]
        print(f"  gripper: {mesh.faces.shape[0]} faces, extent {extent} m")

    print("\nDone! Meshes saved to:")
    print(f"  Visual:    {VISUAL_DIR}/")
    print(f"  Collision: {COLLISION_DIR}/")


if __name__ == "__main__":
    main()
