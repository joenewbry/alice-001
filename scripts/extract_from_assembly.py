#!/usr/bin/env python3
"""Extract per-link meshes from the XArm.STEP assembly file.

The XArm.STEP file contains 41 solids representing the full arm assembly.
This script groups them by spatial position into the robot's kinematic links,
tessellates to STL, scales to meters, and saves visual + collision meshes.

Based on inspection, the arm extends along the Y axis:
- Base: Y ~0-80mm (base platform + turntable)
- Shoulder: Y ~80-140mm (first servo bracket)
- Upper arm: Y ~140-230mm (long link)
- Lower arm: Y ~230-310mm (long link)
- Wrist/Gripper: Y ~310-330mm (end effector)

Solids are grouped by analyzing their Y-center positions against the known
joint locations from the MuJoCo model.
"""

import os
import sys
import tempfile
import trimesh
import fast_simplification
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer

STEP_FILE = "/Users/joe/Downloads/XArm.STEP"
OUTPUT_DIR = os.path.expanduser("~/dev/Alice-001/meshes")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visual")
COLLISION_DIR = os.path.join(OUTPUT_DIR, "collision")

MM_TO_M = 0.001
VISUAL_TARGET_FACES = 10000

# Y-boundaries for grouping solids into links (in mm)
# Derived from solid positions and known kinematic structure
# Looking at the solids:
# Solid 13 (Y=293.5, large bracket) and 15 (Y=309.1) are clearly the wrist area
# Solid 17 (Y=274.3) is a servo horn on the lower arm/wrist boundary
# Adjusting boundaries to put the wrist bracket correctly
LINK_BOUNDARIES = [
    ("base_link",       0,    80),
    ("shoulder_link",   80,  140),
    ("upper_arm_link", 140,  240),
    ("lower_arm_link", 240,  280),
    ("wrist_link",     280,  400),
]


def load_step_solids(path: str) -> list[tuple]:
    """Load STEP and return list of (solid, bbox_center, bbox_extent)."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 1:
        raise RuntimeError(f"Failed to read STEP file: status={status}")

    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solid = exp.Current()
        bbox = Bnd_Box()
        BRepBndLib.Add_s(solid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        center = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
        extent = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
        solids.append((solid, center, extent))
        exp.Next()

    return solids


def group_solids(solids: list[tuple]) -> dict:
    """Group solids into links based on Y-center position."""
    groups = {name: [] for name, _, _ in LINK_BOUNDARIES}

    for solid, center, extent in solids:
        y_center = center[1]
        assigned = False
        for name, y_min, y_max in LINK_BOUNDARIES:
            if y_min <= y_center < y_max:
                groups[name].append((solid, center, extent))
                assigned = True
                break
        if not assigned:
            print(f"  WARNING: Solid at Y={y_center:.1f} not assigned to any link")
            # Assign to nearest link
            min_dist = float('inf')
            nearest = LINK_BOUNDARIES[0][0]
            for name, y_min, y_max in LINK_BOUNDARIES:
                mid = (y_min + y_max) / 2
                dist = abs(y_center - mid)
                if dist < min_dist:
                    min_dist = dist
                    nearest = name
            groups[nearest].append((solid, center, extent))
            print(f"    -> Assigned to {nearest}")

    return groups


def compound_to_trimesh(solids_list, linear_deflection=0.5) -> trimesh.Trimesh:
    """Convert list of OCC solids to a single trimesh."""
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for solid, _, _ in solids_list:
        builder.Add(compound, solid)

    # Tessellate
    mesh_occ = BRepMesh_IncrementalMesh(compound, linear_deflection)
    mesh_occ.Perform()

    # Write to temp STL
    writer = StlAPI_Writer()
    writer.ASCIIMode = False

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

    print("Loading STEP assembly...")
    solids = load_step_solids(STEP_FILE)
    print(f"  Found {len(solids)} solids")

    # Print all solid positions for debugging
    print("\nSolid positions (Y-center):")
    for i, (_, center, extent) in enumerate(solids):
        print(f"  {i:2d}: Y={center[1]:6.1f}  center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})  "
              f"extent=({extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f})")

    print("\nGrouping solids into links...")
    groups = group_solids(solids)

    for name, solids_in_group in groups.items():
        n = len(solids_in_group)
        if n == 0:
            print(f"\n{name}: NO SOLIDS (skipping)")
            continue

        print(f"\n{name}: {n} solids")

        mesh = compound_to_trimesh(solids_in_group)

        # Scale mm -> m
        mesh.apply_scale(MM_TO_M)

        extent = mesh.bounds[1] - mesh.bounds[0]
        print(f"  Tessellated: {mesh.faces.shape[0]} faces, extent {extent} m")

        # Visual
        visual = decimate(mesh, VISUAL_TARGET_FACES)
        print(f"  Visual: {mesh.faces.shape[0]} -> {visual.faces.shape[0]} faces")
        visual.export(os.path.join(VISUAL_DIR, f"{name}.stl"))

        # Collision
        collision = mesh.convex_hull
        print(f"  Collision: {collision.faces.shape[0]} faces")
        collision.export(os.path.join(COLLISION_DIR, f"{name}.stl"))

    print("\n=== Assembly extraction complete! ===")
    print(f"  Visual:    {VISUAL_DIR}/")
    print(f"  Collision: {COLLISION_DIR}/")


if __name__ == "__main__":
    main()
