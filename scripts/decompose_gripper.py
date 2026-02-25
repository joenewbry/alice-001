#!/usr/bin/env python3
"""Decompose gripper STEP files into separate STL meshes for simulation.

Groups individual STEP parts from ~/Downloads/gripper/ into:
- gripper_base: bottom_plate + top_cover + handle
- left_finger: finger_holder_left + linkage_left
- right_finger: finger_holder_right + linkage_right

Outputs scaled (mm -> m), decimated visual and convex hull collision meshes.
"""

import os
import cadquery as cq
import trimesh
import fast_simplification
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.StlAPI import StlAPI_Writer
import tempfile

GRIPPER_DIR = os.path.expanduser("~/Downloads/gripper")
OUTPUT_DIR = os.path.expanduser("~/dev/Alice-001/meshes")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")
COLLISION_DIR = os.path.join(OUTPUT_DIR, "collision")

MM_TO_M = 0.001
VISUAL_TARGET_FACES = 10000

# Part groupings
GROUPS = {
    "gripper_base": [
        "gripper - bottom_plate.step",
        "gripper - top_cover.step",
        "gripper - handle.step",
    ],
    "left_finger": [
        "gripper - finger_holder_left.step",
        "gripper - linkage_left.step",
    ],
    "right_finger": [
        "gripper - finger_holder_right.step",
        "gripper - linkage_right.step",
    ],
}


def load_step_to_compound(step_files: list[str]) -> TopoDS_Compound:
    """Load multiple STEP files and combine into a compound shape."""
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for sf in step_files:
        path = os.path.join(GRIPPER_DIR, sf)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != 1:  # IFSelect_RetDone
            print(f"  WARNING: Failed to read {sf}")
            continue

        reader.TransferRoots()
        shape = reader.OneShape()
        builder.Add(compound, shape)
        print(f"  Loaded: {sf}")

    return compound


def compound_to_stl_trimesh(compound, linear_deflection=0.1) -> trimesh.Trimesh:
    """Convert OCC compound to trimesh via temporary STL."""
    writer = StlAPI_Writer()
    writer.ASCIIMode = False

    # Mesh the shape first
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    mesh = BRepMesh_IncrementalMesh(compound, linear_deflection)
    mesh.Perform()

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp_path = f.name

    writer.Write(compound, tmp_path)
    result = trimesh.load(tmp_path, force="mesh")
    os.unlink(tmp_path)
    return result


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate mesh to approximate target face count."""
    result = mesh.copy()
    for _ in range(5):
        n_faces = result.faces.shape[0]
        if n_faces <= target_faces * 1.2:
            break
        target_reduction = 1.0 - (target_faces / n_faces)
        target_reduction = min(max(target_reduction, 0.01), 0.99)
        verts_out, faces_out = fast_simplification.simplify(
            result.vertices.view(np.ndarray).astype(np.float64),
            result.faces.view(np.ndarray).astype(np.int64),
            target_reduction=target_reduction,
            agg=7,
        )
        result = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
    return result


def main():
    os.makedirs(VISUAL_DIR, exist_ok=True)
    os.makedirs(COLLISION_DIR, exist_ok=True)

    for name, parts in GROUPS.items():
        print(f"\nProcessing {name}...")

        compound = load_step_to_compound(parts)
        mesh = compound_to_stl_trimesh(compound)

        # Scale from mm to meters
        mesh.apply_scale(MM_TO_M)

        extent = mesh.bounds[1] - mesh.bounds[0]
        print(f"  {name}: {mesh.faces.shape[0]} faces, extent {extent} m")

        # Visual
        visual = decimate(mesh, VISUAL_TARGET_FACES)
        print(f"  Visual: {mesh.faces.shape[0]} -> {visual.faces.shape[0]} faces")
        visual.export(os.path.join(VISUAL_DIR, f"{name}.stl"))

        # Collision
        collision = mesh.convex_hull
        print(f"  Collision: {collision.faces.shape[0]} faces (convex hull)")
        collision.export(os.path.join(COLLISION_DIR, f"{name}.stl"))

    print("\nGripper decomposition complete!")


if __name__ == "__main__":
    main()
