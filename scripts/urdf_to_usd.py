#!/usr/bin/env python3
"""Convert Alice 001 URDF to USD using OpenUSD Python API.

This script creates a simulation-ready USD file directly from the URDF,
applying PhysicsArticulationRootAPI, RigidBodyAPI, CollisionAPI, MassAPI,
and DriveAPI without requiring Isaac Sim or a GPU.

When loaded into Isaac Lab, the robot will be fully articulated with:
- Fixed base (ArticulationRoot on base_link)
- All 7 revolute joints with drive APIs
- Collision meshes on all links
- Mass/inertia from URDF
"""

import os
import math
import xml.etree.ElementTree as ET
import numpy as np

from pxr import (
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    Sdf,
    Gf,
    Vt,
    Kind,
)

PROJECT_DIR = os.path.expanduser("~/dev/Alice-001")
URDF_PATH = os.path.join(PROJECT_DIR, "urdf", "alice_001.urdf")
USD_PATH = os.path.join(PROJECT_DIR, "usd", "alice_001.usd")


def parse_xyz(text):
    """Parse 'x y z' string to Gf.Vec3d."""
    if text is None:
        return Gf.Vec3d(0, 0, 0)
    parts = [float(x) for x in text.split()]
    return Gf.Vec3d(*parts)


def parse_rpy(text):
    """Parse 'r p y' string to Gf.Quatf."""
    if text is None:
        return Gf.Quatf(1, 0, 0, 0)
    r, p, y = [float(x) for x in text.split()]
    # RPY to quaternion
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(w, x, y_, z)


def load_stl_as_mesh(stl_path):
    """Load binary STL and return vertices, face_indices, face_counts."""
    import struct

    with open(stl_path, "rb") as f:
        header = f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]

        vertices = []
        indices = []

        # Use a dict for vertex deduplication
        vertex_map = {}
        idx = 0

        for i in range(num_triangles):
            normal = struct.unpack("<3f", f.read(12))
            v1 = struct.unpack("<3f", f.read(12))
            v2 = struct.unpack("<3f", f.read(12))
            v3 = struct.unpack("<3f", f.read(12))
            attr = struct.unpack("<H", f.read(2))

            tri_indices = []
            for v in [v1, v2, v3]:
                # Round to avoid floating point issues
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in vertex_map:
                    vertex_map[key] = idx
                    vertices.append(Gf.Vec3f(*v))
                    idx += 1
                tri_indices.append(vertex_map[key])
            indices.extend(tri_indices)

    face_counts = [3] * num_triangles
    return vertices, indices, face_counts


def create_material(stage, path, color):
    """Create a simple display material."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_mesh_to_prim(stage, parent_path, mesh_path, name, material=None):
    """Load STL and add as UsdGeom.Mesh under parent."""
    if not os.path.exists(mesh_path):
        print(f"  WARNING: {mesh_path} not found, skipping mesh")
        return None

    vertices, indices, face_counts = load_stl_as_mesh(mesh_path)

    mesh_prim_path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(vertices))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
    mesh.CreateSubdivisionSchemeAttr("none")

    if material:
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    return mesh


def main():
    os.makedirs(os.path.dirname(USD_PATH), exist_ok=True)

    # Parse URDF
    tree = ET.parse(URDF_PATH)
    urdf_root = tree.getroot()

    links = {}
    for link_el in urdf_root.findall("link"):
        name = link_el.attrib["name"]
        links[name] = link_el

    joints = []
    for joint_el in urdf_root.findall("joint"):
        joints.append(joint_el)

    # Build parent->children map
    parent_map = {}  # child_name -> (joint_el, parent_name)
    children_map = {}  # parent_name -> [(joint_el, child_name)]
    for joint_el in joints:
        parent_name = joint_el.find("parent").attrib["link"]
        child_name = joint_el.find("child").attrib["link"]
        parent_map[child_name] = (joint_el, parent_name)
        children_map.setdefault(parent_name, []).append((joint_el, child_name))

    # Find root link (no parent)
    root_link = None
    for link_name in links:
        if link_name not in parent_map:
            root_link = link_name
            break
    print(f"Root link: {root_link}")

    # Create USD stage
    stage = Usd.Stage.CreateNew(USD_PATH)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    # Set default prim
    root_path = "/alice_001"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root_xform.GetPrim())
    Usd.ModelAPI(root_xform.GetPrim()).SetKind(Kind.Tokens.component)

    # Apply ArticulationRootAPI to the root
    UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

    # Create materials
    mat_dark = create_material(stage, f"{root_path}/Looks/DarkGrey", (0.3, 0.3, 0.3))
    mat_blue = create_material(stage, f"{root_path}/Looks/Blue", (0.2, 0.4, 0.8))
    mat_yellow = create_material(stage, f"{root_path}/Looks/Yellow", (0.9, 0.8, 0.2))

    material_map = {
        "dark_grey": mat_dark,
        "blue": mat_blue,
        "yellow": mat_yellow,
    }

    # Colors per link from URDF
    link_colors = {}
    for link_el in urdf_root.findall("link"):
        name = link_el.attrib["name"]
        visual = link_el.find("visual")
        if visual is not None:
            mat = visual.find("material")
            if mat is not None:
                link_colors[name] = mat.attrib.get("name", "dark_grey")

    # Compute world transforms for each link (cumulative from root)
    world_transforms = {}  # link_name -> (world_xyz as Gf.Vec3d, world_rpy_str or None)

    def compute_world_transforms(link_name, parent_world_xyz=Gf.Vec3d(0, 0, 0)):
        """Compute world-frame position for each link."""
        world_transforms[link_name] = parent_world_xyz
        for child_joint_el, child_name in children_map.get(link_name, []):
            origin = child_joint_el.find("origin")
            xyz = parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
            child_world = Gf.Vec3d(
                parent_world_xyz[0] + xyz[0],
                parent_world_xyz[1] + xyz[1],
                parent_world_xyz[2] + xyz[2],
            )
            compute_world_transforms(child_name, child_world)

    compute_world_transforms(root_link)

    # Build FLAT hierarchy: all links as direct children of root_path
    # This avoids the "nested RigidBody" error in PhysX
    link_paths = {}  # link_name -> usd_path

    def build_link(link_name):
        link_el = links[link_name]
        link_path = f"{root_path}/{link_name}"
        link_paths[link_name] = link_path

        link_xform = UsdGeom.Xform.Define(stage, link_path)

        # Set initial world-frame position so bodies aren't disjointed
        world_pos = world_transforms[link_name]
        link_xform.AddTranslateOp().Set(world_pos)

        # Apply RigidBodyAPI (no kinematic on root — we'll use FixedJoint instead)
        UsdPhysics.RigidBodyAPI.Apply(link_xform.GetPrim())

        # Mass/inertia
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            origin_el = inertial.find("origin")
            inertia_el = inertial.find("inertia")

            mass_api = UsdPhysics.MassAPI.Apply(link_xform.GetPrim())
            if mass_el is not None:
                mass_api.CreateMassAttr(float(mass_el.attrib["value"]))
            if origin_el is not None:
                com = parse_xyz(origin_el.attrib.get("xyz"))
                mass_api.CreateCenterOfMassAttr(com)
            if inertia_el is not None:
                ixx = float(inertia_el.attrib.get("ixx", 0))
                iyy = float(inertia_el.attrib.get("iyy", 0))
                izz = float(inertia_el.attrib.get("izz", 0))
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, iyy, izz))

        # Add visual mesh
        color_name = link_colors.get(link_name, "dark_grey")
        material = material_map.get(color_name, mat_dark)

        visual_stl = os.path.join(PROJECT_DIR, "meshes", "visual", f"{link_name}.stl")
        visual_el = link_el.find("visual")
        visual_origin = None
        if visual_el is not None:
            vo = visual_el.find("origin")
            if vo is not None:
                visual_origin = parse_xyz(vo.attrib.get("xyz"))

        visual_mesh = add_mesh_to_prim(
            stage, link_path, visual_stl, "visual", material
        )
        if visual_mesh and visual_origin:
            visual_mesh.AddTranslateOp().Set(visual_origin)

        # Add collision mesh
        collision_stl = os.path.join(PROJECT_DIR, "meshes", "collision", f"{link_name}.stl")
        collision_el = link_el.find("collision")
        collision_origin = None
        if collision_el is not None:
            co = collision_el.find("origin")
            if co is not None:
                collision_origin = parse_xyz(co.attrib.get("xyz"))

        col_mesh = add_mesh_to_prim(
            stage, link_path, collision_stl, "collision"
        )
        if col_mesh:
            UsdPhysics.CollisionAPI.Apply(col_mesh.GetPrim())
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(col_mesh.GetPrim())
            mesh_col.CreateApproximationAttr("convexHull")
            col_mesh.CreateVisibilityAttr("invisible")
            if collision_origin:
                col_mesh.AddTranslateOp().Set(collision_origin)

        print(f"  Link: {link_name}")

    # Build all links as flat children
    print("\nBuilding USD hierarchy (flat)...")
    for link_name in links:
        build_link(link_name)

    # Create a FixedJoint to anchor base_link to the world
    # (PhysX doesn't support kinematic bodies in articulations)
    fixed_joint_path = f"{root_path}/fixed_base_joint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
    # body0 = None (world), body1 = base_link
    fixed_joint.CreateBody1Rel().SetTargets([link_paths[root_link]])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    print("  Fixed joint: base_link -> world")

    # Now create revolute joints as children of root_path
    print("\nCreating joints...")
    for joint_el in joints:
        joint_name = joint_el.attrib["name"]
        joint_type = joint_el.attrib["type"]
        parent_name = joint_el.find("parent").attrib["link"]
        child_name = joint_el.find("child").attrib["link"]
        origin = joint_el.find("origin")
        xyz = parse_xyz(origin.attrib.get("xyz") if origin is not None else None)

        parent_path = link_paths[parent_name]
        child_path = link_paths[child_name]

        if joint_type == "revolute":
            axis_el = joint_el.find("axis")
            axis = [float(x) for x in axis_el.attrib["xyz"].split()]
            limit_el = joint_el.find("limit")

            # Determine USD joint axis
            if abs(axis[2]) > 0.5:
                usd_axis = "Z"
            elif abs(axis[1]) > 0.5:
                usd_axis = "Y"
            else:
                usd_axis = "X"

            joint_path = f"{root_path}/{joint_name}"
            rev_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

            rev_joint.CreateAxisAttr(usd_axis)

            # Set body references
            rev_joint.CreateBody0Rel().SetTargets([parent_path])
            rev_joint.CreateBody1Rel().SetTargets([child_path])

            # Local transforms: joint position relative to each body
            # localPos0 = joint position in parent frame = the URDF joint origin xyz
            rev_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            rev_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
            # localPos1 = joint position in child frame = (0,0,0) since joint is at child origin
            rev_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            rev_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

            # Joint limits
            if limit_el is not None:
                lo = float(limit_el.attrib.get("lower", 0))
                hi = float(limit_el.attrib.get("upper", 0))
                lo_deg = math.degrees(lo)
                hi_deg = math.degrees(hi)
                rev_joint.CreateLowerLimitAttr(lo_deg)
                rev_joint.CreateUpperLimitAttr(hi_deg)

            # Add drive API for actuation
            drive_api = UsdPhysics.DriveAPI.Apply(
                rev_joint.GetPrim(), usd_axis
            )
            drive_api.CreateTypeAttr("force")

            if "finger" in joint_name:
                stiffness = 20.0
                damping = 2.0
                max_force = 1.0
            else:
                stiffness = 40.0
                damping = 4.0
                max_force = 2.5

            drive_api.CreateStiffnessAttr(stiffness)
            drive_api.CreateDampingAttr(damping)
            drive_api.CreateMaxForceAttr(max_force)

            lo_deg_str = f"{math.degrees(float(limit_el.attrib.get('lower', 0))):.1f}" if limit_el is not None else "?"
            hi_deg_str = f"{math.degrees(float(limit_el.attrib.get('upper', 0))):.1f}" if limit_el is not None else "?"
            print(f"  Joint: {joint_name} ({usd_axis} axis, [{lo_deg_str}, {hi_deg_str}] deg)")

    # Enable physics scene
    physics_scene_path = f"{root_path}/PhysicsScene"
    scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    # Save
    stage.Save()

    # Also export as USDA for inspection
    usda_path = USD_PATH.replace(".usd", ".usda")
    stage.Export(usda_path)

    print(f"\nUSD saved: {USD_PATH}")
    print(f"USDA saved: {usda_path}")

    # Print summary
    print(f"\nSummary:")
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
    print(f"  Total prims: {prim_count}")
    print(f"  Links: {len(links)}")
    print(f"  Joints: {len(joints)}")


if __name__ == "__main__":
    main()
