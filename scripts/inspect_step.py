#!/usr/bin/env python3
"""Inspect STEP assembly files to understand part hierarchy."""

import sys
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def inspect_step(path: str):
    print(f"\n=== Inspecting: {path} ===")
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 1:
        print(f"  ERROR: Failed to read (status={status})")
        return

    # Transfer all roots
    n_roots = reader.NbRootsForTransfer()
    print(f"  Roots: {n_roots}")
    reader.TransferRoots()

    n_shapes = reader.NbShapes()
    print(f"  Shapes: {n_shapes}")

    # Explore top-level solids
    shape = reader.OneShape()

    # Count different topology types
    for type_name, type_val in [
        ("COMPOUND", TopAbs_COMPOUND),
        ("SOLID", TopAbs_SOLID),
        ("SHELL", TopAbs_SHELL),
        ("FACE", TopAbs_FACE),
    ]:
        exp = TopExp_Explorer(shape, type_val)
        count = 0
        while exp.More():
            count += 1
            exp.Next()
        print(f"  {type_name}: {count}")

    # Bounding box
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"  Bounding box (mm):")
    print(f"    X: {xmin:.1f} to {xmax:.1f} ({xmax-xmin:.1f} mm)")
    print(f"    Y: {ymin:.1f} to {ymax:.1f} ({ymax-ymin:.1f} mm)")
    print(f"    Z: {zmin:.1f} to {zmax:.1f} ({zmax-zmin:.1f} mm)")

    # Print individual solid bounding boxes (first 20)
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    i = 0
    while exp.More() and i < 20:
        solid = exp.Current()
        sb = Bnd_Box()
        BRepBndLib.Add_s(solid, sb)
        sx0, sy0, sz0, sx1, sy1, sz1 = sb.Get()
        extent = f"({sx1-sx0:.1f} x {sy1-sy0:.1f} x {sz1-sz0:.1f})"
        print(f"    Solid {i}: center=({(sx0+sx1)/2:.1f}, {(sy0+sy1)/2:.1f}, {(sz0+sz1)/2:.1f}) extent={extent}")
        exp.Next()
        i += 1


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "/Users/joe/Downloads/XArm.STEP",
        "/Users/joe/Downloads/xArm 1S.stp",
    ]
    for f in files:
        inspect_step(f)
