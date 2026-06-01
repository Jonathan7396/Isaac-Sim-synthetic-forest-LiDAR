"""
Isaac Sim 5.1 RTX LiDAR semantic point cloud capture.

This script uses an already-created VLS-128 LiDAR asset in the Isaac Sim scene,
captures point clouds from a fixed sensor position, resolves RTX object IDs to USD
prim paths using StableIdMap, and saves XYZ, object IDs, prim paths, semantic labels,
intensity values, and a CSV export.

Required Isaac Sim launch flag:
    --/rtx-transient/stableIds/enabled=True

Scene convention:
    - Y is the up axis.
    - The LiDAR rig is moved using xformOp:translate and xformOp:rotateXYZ.
    - Semantic labels may be attached directly to a prim or to one of its parents.
"""

import asyncio
import os
from collections import Counter

import omni.kit.app
import omni.timeline
import omni.replicator.core as rep
import omni.usd

import numpy as np
from pxr import Gf, Semantics, UsdGeom

from isaacsim.sensors.rtx import LidarRtx


# -----------------------------------------------------------------------------
# Scene handles
# -----------------------------------------------------------------------------

stage = omni.usd.get_context().get_stage()
app = omni.kit.app.get_app()
timeline = omni.timeline.get_timeline_interface()


# -----------------------------------------------------------------------------
# LiDAR asset paths
# -----------------------------------------------------------------------------

# Parent rig/Xform that should be moved in the scene.
LIDAR_RIG_PATH = "/World/VLS_128"

# Actual OmniLidar sensor inside the VLS rig.
LIDAR_SENSOR_PATH = "/World/VLS_128/Ouster_VLS_128"

# Keep the rotation that is already working in your scene.
LIDAR_EULER_DEG = Gf.Vec3d(0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Fixed-position capture configuration
# -----------------------------------------------------------------------------

# Y is the up axis in this scene.
# Change this one value when you want to move the LiDAR.
# Format: Gf.Vec3d(X, Y, Z)
LIDAR_POSITION = Gf.Vec3d(297.0, 23.0, -5551.0)

# The capture loop still expects a list of positions.
# For the current fixed-position workflow, keep this as one position only.
CAPTURE_POSITIONS = [LIDAR_POSITION]

# Fixed-position repeated capture setup.
# This keeps the old-style console output:
#     Step 1/100
#     Step points: ...
#     Step label counts: ...
#
# The LiDAR does not move between these steps. Each visible step captures
# multiple internal frames from the same fixed position, then merges them.
#
# Equivalent to the old setup:
#     NUM_Z_LANES = 5
#     STEPS_PER_SWATH = 20
#     SUBFRAMES_PER_STEP = 4
#     5 * 20 = 100 visible steps
#     100 * 4 = 400 total LiDAR frame captures
STEPS_PER_POSITION = 100
FRAMES_PER_STEP = 4
FRAME_DELAY_SEC = 0.02

RENDER_RESOLUTION = (512, 512)

STARTUP_DELAY_SEC = 3.0
WARMUP_UPDATES = 30

OUTPUT_DIR = os.path.abspath("lidar_output_manual_VLS128_asset_semantic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENABLE_DEBUG_DRAW = False
SAVE_DEBUG_CLOUD = True

# Print old-style step progress so the console does not look frozen.
PRINT_STEP_PROGRESS = True

# Print semantic label counts after every visible step.
# Keep this enabled when checking whether tree labels are being captured correctly.
PRINT_STEP_LABEL_COUNTS = True

# Set to an integer, such as 20, if the console output becomes too long.
# Keep as None to print all labels found in each step.
MAX_LABELS_TO_PRINT_PER_STEP = None


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def log_info(message):
    print(f"[INFO] {message}")


def log_warning(message):
    print(f"[WARNING] {message}")


def log_error(message):
    print(f"[ERROR] {message}")


# -----------------------------------------------------------------------------
# USD transform helpers
# -----------------------------------------------------------------------------

def get_or_create_translate_op(prim):
    attr = prim.GetAttribute("xformOp:translate")
    if attr.IsValid():
        return attr

    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op.GetAttr()

    return xf.AddTranslateOp().GetAttr()


def get_or_create_rotate_xyz_op(prim):
    attr = prim.GetAttribute("xformOp:rotateXYZ")
    if attr.IsValid():
        return attr

    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            return op.GetAttr()

    return xf.AddRotateXYZOp().GetAttr()


def coerce_vec3_for_attr(attr, vec):
    """Return Vec3f or Vec3d depending on the USD attribute type."""
    try:
        type_name = str(attr.GetTypeName()).lower()
        if "float3" in type_name or "vec3f" in type_name:
            return Gf.Vec3f(float(vec[0]), float(vec[1]), float(vec[2]))
    except Exception:
        pass

    return Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2]))


def set_xform_order_translate_rotate(prim):
    """
    Force a clean transform order for the LiDAR rig.

    This avoids xformOpOrder warnings caused by old scale, matrix, or extra ops.
    """
    order_attr = prim.GetAttribute("xformOpOrder")
    if order_attr.IsValid():
        order_attr.Set(["xformOp:translate", "xformOp:rotateXYZ"])


def set_pose_euler(prim, position, euler_deg):
    translate_attr = get_or_create_translate_op(prim)
    rotate_attr = get_or_create_rotate_xyz_op(prim)

    translate_attr.Set(coerce_vec3_for_attr(translate_attr, position))
    rotate_attr.Set(coerce_vec3_for_attr(rotate_attr, euler_deg))

    set_xform_order_translate_rotate(prim)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def clean_xyz(xyz):
    """Return a valid N x 3 float32 XYZ array."""
    if xyz is None:
        return np.empty((0, 3), dtype=np.float32)

    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)

    xyz = xyz[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    if xyz.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    return xyz


def get_semantic_label_from_prim_path(stage, prim_path):
    """
    Resolve a semantic label from a USD prim path.

    The search walks up the prim hierarchy because labels are often stored on a
    parent prim rather than on the exact mesh prim hit by the LiDAR ray.
    """
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        return "unmapped"

    while prim.IsValid():
        # 1. Legacy SemanticsAPI path.
        try:
            sem_api = Semantics.SemanticsAPI.Get(prim, "class")
            if sem_api:
                data_attr = sem_api.GetSemanticDataAttr()
                if data_attr and data_attr.HasValue():
                    value = data_attr.Get()
                    if value:
                        return str(value)
        except Exception:
            pass

        # 2. Newer SemanticsLabelsAPI-style attribute path.
        try:
            attr = prim.GetAttribute("semantics:labels:class")
            if attr and attr.IsValid() and attr.HasValue():
                value = attr.Get()
                if value:
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        return str(value[0])
                    return str(value)
        except Exception:
            pass

        # 3. Fallback scan for semantic-like attributes.
        try:
            for prop in prim.GetProperties():
                name = prop.GetName().lower()
                if "semantic" in name:
                    try:
                        value = prop.Get()
                        if value:
                            if isinstance(value, (list, tuple)) and len(value) > 0:
                                return str(value[0])
                            return str(value)
                    except Exception:
                        pass
        except Exception:
            pass

        prim = prim.GetParent()

    return "unlabeled"


def save_mapping_debug(mapping_dict, path, limit=500):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Stable ID map entries: {len(mapping_dict)}\n\n")

        for idx, (stable_id, prim_path) in enumerate(mapping_dict.items()):
            if idx >= limit:
                f.write("\n... truncated ...\n")
                break

            f.write(f"{stable_id} -> {prim_path}\n")


def decode_stable_id_map(stable_id_annotator):
    """Read and decode StableIdMap data from the annotator."""
    stable_map_raw = stable_id_annotator.get_data()
    log_info(f"StableIdMap raw type: {type(stable_map_raw)}")

    if isinstance(stable_map_raw, dict):
        log_info(f"StableIdMap dict keys: {list(stable_map_raw.keys())}")
        if "data" not in stable_map_raw:
            raise RuntimeError(
                "StableIdMap returned a dictionary without a 'data' key. "
                f"Keys: {list(stable_map_raw.keys())}"
            )
        stable_map_payload = stable_map_raw["data"]
    else:
        stable_map_payload = stable_map_raw

    if isinstance(stable_map_payload, np.ndarray):
        stable_map_payload = stable_map_payload.tobytes()

    return LidarRtx.decode_stable_id_mapping(stable_map_payload)


def safe_intensity_array(raw_intensity, expected_length):
    if raw_intensity is None:
        return np.zeros(expected_length, dtype=np.float32)

    intensity = np.asarray(raw_intensity, dtype=np.float32)

    if len(intensity) != expected_length:
        return np.zeros(expected_length, dtype=np.float32)

    return intensity


def format_label_counts(labels, max_labels=None):
    counts = Counter(labels)
    items = counts.most_common(max_labels)

    formatted = "{" + ", ".join(
        f"'{label}': {count}" for label, count in items
    ) + "}"

    if max_labels is not None and len(counts) > max_labels:
        formatted += f"  ... plus {len(counts) - max_labels} more labels"

    return formatted


def print_label_counts(labels, heading):
    counts = Counter(labels)
    log_info(heading)
    for label, count in counts.most_common():
        print(f"    {label}: {count}")


def log_step_progress(step_index, total_steps, message):
    if not PRINT_STEP_PROGRESS:
        return

    print(flush=True)
    print(
        f"--- Step {step_index + 1}/{total_steps} | {message} ---",
        flush=True,
    )


def save_outputs(merged_xyz, all_u128_ids, all_prim_paths, all_labels, all_intensity):
    """Save all output files while preserving the existing file names."""
    log_info(f"Total merged XYZ points: {merged_xyz.shape[0]}")
    log_info(f"Total labels: {len(all_labels)}")
    print("XYZ min:", merged_xyz.min(axis=0))
    print("XYZ max:", merged_xyz.max(axis=0))
    print("XYZ mean:", merged_xyz.mean(axis=0))

    if SAVE_DEBUG_CLOUD:
        np.save(os.path.join(OUTPUT_DIR, "y_up_xyz.npy"), merged_xyz)

    np.save(
        os.path.join(OUTPUT_DIR, "y_up_object_ids_u128.npy"),
        np.array(all_u128_ids, dtype=object),
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_up_prim_paths.npy"),
        np.array(all_prim_paths, dtype=object),
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_up_semantic_labels.npy"),
        np.array(all_labels, dtype=object),
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_up_intensity.npy"),
        np.array(all_intensity, dtype=np.float32),
    )

    csv_path = os.path.join(OUTPUT_DIR, "y_up_semantic_points.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("x,y,z,intensity,object_id_u128,prim_path,semantic_label\n")

        for i in range(merged_xyz.shape[0]):
            x, y, z = merged_xyz[i]
            intensity = all_intensity[i]
            obj_id = all_u128_ids[i]
            prim_path = str(all_prim_paths[i]).replace(",", ";")
            label = str(all_labels[i]).replace(",", ";")

            f.write(f"{x},{y},{z},{intensity},{obj_id},{prim_path},{label}\n")

    log_info(f"Saved semantic CSV: {csv_path}")
    print_label_counts(all_labels, "Final label counts:")


# -----------------------------------------------------------------------------
# LiDAR configuration inspection
# -----------------------------------------------------------------------------

def inspect_lidar_config(sensor_prim):
    print("\n================ MANUAL VLS SENSOR CHECK ================")
    print("Rig path:", LIDAR_RIG_PATH)
    print("Sensor path:", sensor_prim.GetPath())
    print("Sensor type:", sensor_prim.GetTypeName())
    print("Applied schemas:", sensor_prim.GetAppliedSchemas())

    attrs_to_check = [
        "omni:sensor:marketName",
        "omni:sensor:modelName",
        "omni:sensor:modelVendor",
        "omni:sensor:modelVersion",
        "omni:sensor:Core:numberOfChannels",
        "omni:sensor:Core:numberOfEmitters",
        "omni:sensor:Core:scanType",
        "omni:sensor:Core:validStartAzimuthDeg",
        "omni:sensor:Core:validEndAzimuthDeg",
        "omni:sensor:Core:nearRangeM",
        "omni:sensor:Core:farRangeM",
        "omni:sensor:Core:scanRateBaseHz",
        "omni:sensor:tickRate",
        "omni:sensor:Core:outputFrameOfReference",
        "omni:sensor:Core:auxOutputType",
    ]

    for attr_name in attrs_to_check:
        attr = sensor_prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            try:
                print(f"{attr_name}: {attr.Get()}")
            except Exception:
                print(f"{attr_name}: <could not read>")

    elev_attr = sensor_prim.GetAttribute(
        "omni:sensor:Core:emitterState:s001:elevationDeg"
    )
    if elev_attr and elev_attr.IsValid():
        vals = elev_attr.Get()
        if vals:
            vals = list(vals)
            print("Elevation count:", len(vals))
            print("Elevation min/max:", min(vals), max(vals))
            print("First 16 elevations:", vals[:16])

    print("==========================================================\n")


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def validate_lidar_prims():
    rig_prim = stage.GetPrimAtPath(LIDAR_RIG_PATH)
    sensor_prim = stage.GetPrimAtPath(LIDAR_SENSOR_PATH)

    if not rig_prim.IsValid():
        raise RuntimeError(f"LiDAR rig not found: {LIDAR_RIG_PATH}")

    if not sensor_prim.IsValid():
        raise RuntimeError(f"LiDAR sensor not found: {LIDAR_SENSOR_PATH}")

    if sensor_prim.GetTypeName() != "OmniLidar":
        raise RuntimeError(
            f"Expected OmniLidar at {LIDAR_SENSOR_PATH}, "
            f"got type {sensor_prim.GetTypeName()}"
        )

    return rig_prim, sensor_prim


def configure_sensor_aux_output(sensor_prim):
    aux_attr = sensor_prim.GetAttribute("omni:sensor:Core:auxOutputType")
    if aux_attr.IsValid():
        aux_attr.Set("FULL")
        log_info("Set sensor auxOutputType = FULL")
    else:
        log_warning("auxOutputType attribute not found on sensor")


def create_render_product_and_annotators():
    # The render product must attach to the actual OmniLidar sensor, not the rig.
    render_product = rep.create.render_product(LIDAR_SENSOR_PATH, RENDER_RESOLUTION)

    scan_annotator = rep.AnnotatorRegistry.get_annotator(
        "IsaacCreateRTXLidarScanBuffer"
    )
    scan_annotator.initialize(
        outputDistance=True,
        outputIntensity=True,
        outputTimestamp=True,
        outputMaterialId=False,
        outputObjectId=True,
        transformPoints=True,
    )
    scan_annotator.attach([render_product.path])

    stable_id_annotator = rep.AnnotatorRegistry.get_annotator("StableIdMap")
    stable_id_annotator.attach([render_product.path])

    log_info(f"Attached LiDAR scan annotator to {LIDAR_SENSOR_PATH}")
    log_info("Attached StableIdMap annotator")

    return render_product, scan_annotator, stable_id_annotator


async def attach_debug_draw(render_product):
    await asyncio.sleep(0.5)

    if ENABLE_DEBUG_DRAW:
        writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer.attach([render_product.path])
        log_info("Debug draw enabled")


# -----------------------------------------------------------------------------
# Capture
# -----------------------------------------------------------------------------

async def capture_position(
    position_index,
    position,
    total_positions,
    rig_prim,
    scan_annotator,
    stable_id_to_prim,
):
    """Capture LiDAR subframes at one rig position."""
    set_pose_euler(rig_prim, position, LIDAR_EULER_DEG)

    step_xyz = []
    step_u128 = []
    step_paths = []
    step_labels = []
    step_intensity = []

    print()
    print(
        f"--- Position {position_index + 1}/{total_positions} | "
        f"X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f} ---"
    )

    for step_idx in range(STEPS_PER_POSITION):
        log_step_progress(
            step_idx,
            STEPS_PER_POSITION,
            f"Position {position_index + 1}/{total_positions} | "
            f"X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}",
        )

        visible_step_xyz = []
        visible_step_u128 = []
        visible_step_paths = []
        visible_step_labels = []
        visible_step_intensity = []

        for frame_idx in range(FRAMES_PER_STEP):
            await app.next_update_async()
            await asyncio.sleep(FRAME_DELAY_SEC)

            data = scan_annotator.get_data()
            if not data:
                continue

            pts = data.get("data", None)
            raw_object_ids = data.get("objectId", None)
            raw_intensity = data.get("intensity", None)

            if not isinstance(pts, np.ndarray) or pts.shape[0] == 0:
                continue

            xyz = clean_xyz(pts)
            if xyz.shape[0] == 0:
                continue

            try:
                obj_ids_u128 = LidarRtx.get_object_ids(raw_object_ids)
            except Exception as exc:
                log_warning(
                    f"Failed to decode object IDs at step {step_idx}, "
                    f"frame {frame_idx}: {exc}"
                )
                continue

            if len(obj_ids_u128) != xyz.shape[0]:
                log_warning("objectId/XYZ length mismatch")
                continue

            intensity = safe_intensity_array(raw_intensity, xyz.shape[0])

            prim_paths = []
            labels = []

            for obj_id in obj_ids_u128:
                prim_path = stable_id_to_prim.get(int(obj_id), "")
                prim_paths.append(prim_path)
                labels.append(
                    get_semantic_label_from_prim_path(stage, prim_path)
                    if prim_path
                    else "unmapped"
                )

            visible_step_xyz.append(xyz)
            visible_step_u128.extend([int(x) for x in obj_ids_u128])
            visible_step_paths.extend(prim_paths)
            visible_step_labels.extend(labels)
            visible_step_intensity.extend([float(x) for x in intensity])

        if not visible_step_xyz:
            log_warning("No valid points captured in this step")
            continue

        merged_visible_step_xyz = np.vstack(visible_step_xyz).astype(np.float32)

        if merged_visible_step_xyz.shape[0] != len(visible_step_u128):
            log_warning("Step point/object ID mismatch. Skipping this step")
            continue

        step_xyz.append(merged_visible_step_xyz)
        step_u128.extend(visible_step_u128)
        step_paths.extend(visible_step_paths)
        step_labels.extend(visible_step_labels)
        step_intensity.extend(visible_step_intensity)

        print(f"Step points: {merged_visible_step_xyz.shape[0]}", flush=True)

        if PRINT_STEP_LABEL_COUNTS:
            print(
                "Step label counts:",
                format_label_counts(
                    visible_step_labels,
                    MAX_LABELS_TO_PRINT_PER_STEP,
                ),
                flush=True,
            )

    if not step_xyz:
        log_warning("No valid points captured at this position")
        return None

    merged_step_xyz = np.vstack(step_xyz).astype(np.float32)

    if merged_step_xyz.shape[0] != len(step_u128):
        log_warning("Position point/object ID mismatch. Skipping this position")
        return None

    log_info(f"Position points: {merged_step_xyz.shape[0]}")
    print_label_counts(step_labels, "Position label counts:")

    return {
        "xyz": merged_step_xyz,
        "object_ids": step_u128,
        "prim_paths": step_paths,
        "labels": step_labels,
        "intensity": step_intensity,
    }


async def capture():
    log_info("Capturing LiDAR frames with semantic label resolution")
    await asyncio.sleep(STARTUP_DELAY_SEC)

    for _ in range(WARMUP_UPDATES):
        await app.next_update_async()

    stable_id_to_prim = decode_stable_id_map(stable_id_annotator)

    debug_map_path = os.path.join(OUTPUT_DIR, "stable_id_map_debug.txt")
    save_mapping_debug(stable_id_to_prim, debug_map_path)

    log_info(f"Saved StableIdMap debug: {debug_map_path}")
    log_info(f"StableIdMap entries: {len(stable_id_to_prim)}")

    all_xyz = []
    all_u128_ids = []
    all_prim_paths = []
    all_labels = []
    all_intensity = []

    try:
        for position_index, position in enumerate(CAPTURE_POSITIONS):
            result = await capture_position(
                position_index=position_index,
                position=position,
                total_positions=len(CAPTURE_POSITIONS),
                rig_prim=rig_prim,
                scan_annotator=scan_annotator,
                stable_id_to_prim=stable_id_to_prim,
            )

            if result is None:
                continue

            all_xyz.append(result["xyz"])
            all_u128_ids.extend(result["object_ids"])
            all_prim_paths.extend(result["prim_paths"])
            all_labels.extend(result["labels"])
            all_intensity.extend(result["intensity"])

        if all_xyz:
            merged_xyz = np.vstack(all_xyz).astype(np.float32)
            save_outputs(
                merged_xyz=merged_xyz,
                all_u128_ids=all_u128_ids,
                all_prim_paths=all_prim_paths,
                all_labels=all_labels,
                all_intensity=all_intensity,
            )
        else:
            log_error("No valid labeled points captured")

    finally:
        timeline.stop()
        log_info("Capture complete")
        log_info(f"Output directory: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

log_info("Using manually added VLS_128 LiDAR asset")
log_info(f"Total capture positions: {len(CAPTURE_POSITIONS)}")
log_info(f"Visible steps per position: {STEPS_PER_POSITION}")
log_info(f"Internal frames per visible step: {FRAMES_PER_STEP}")
log_info(
    f"Maximum captured frames: "
    f"{len(CAPTURE_POSITIONS) * STEPS_PER_POSITION * FRAMES_PER_STEP}"
)
log_info(
    f"Capture position: X={LIDAR_POSITION[0]}, "
    f"Y={LIDAR_POSITION[1]}, Z={LIDAR_POSITION[2]}"
)
log_info(f"Rig path: {LIDAR_RIG_PATH}")
log_info(f"Sensor path: {LIDAR_SENSOR_PATH}")

rig_prim, sensor_prim = validate_lidar_prims()

log_info(f"LiDAR rig found: {rig_prim.GetPath()}")
log_info(f"LiDAR sensor found: {sensor_prim.GetPath()}")

set_pose_euler(rig_prim, LIDAR_POSITION, LIDAR_EULER_DEG)
configure_sensor_aux_output(sensor_prim)
inspect_lidar_config(sensor_prim)

rp, scan_annotator, stable_id_annotator = create_render_product_and_annotators()
asyncio.ensure_future(attach_debug_draw(rp))

timeline.play()
log_info("Simulation started")

asyncio.ensure_future(capture())
