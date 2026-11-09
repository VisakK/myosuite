# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Import curated yoga references (ProtoMotions ``.motion`` files).

A ``.motion`` file is a ``torch.save`` dict whose ``state_conversion`` entry is a
ProtoMotions enum. It is read here with a stub unpickler, so ProtoMotions does
not need to be installed. The references live on the performer's 24-body
"plant v2" skeleton (SMPL-X joint centres of her personal template); body order
is the plant MJCF's, rotations are stored ``xyzw`` (verified by forward
kinematics against the plant MJCF to 2e-7 m).

CLI::

    python -m myosuite.integrations.yoga.reference_io \
        --release <release dir with motions/> --plant-xml <smpl_yogi03596_v2.xml> \
        --out <dir>
"""

from __future__ import annotations

import argparse
import pickle
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

PLANT_BODIES: tuple[str, ...] = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle",
    "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Hand",
)  # fmt: skip

# Dwell variants (``*_x3s``/``*_x7s``) repeat a hold; they are not separate motions.
_DWELL_SUFFIXES = ("_x3s", "_x7s")


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that replaces every ProtoMotions class by a tuple placeholder."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("protomotions"):
            return lambda *args, **kwargs: (module, name, args)
        return super().find_class(module, name)


_STUB_PICKLE = types.SimpleNamespace(
    Unpickler=_StubUnpickler,
    load=lambda f, **kw: _StubUnpickler(f, **kw).load(),
    __name__="yoga_stub_pickle",
)


@dataclass(frozen=True)
class YogaReference:
    """One reference motion on the plant skeleton (world frame, z up, floor at 0).

    Attributes:
        name: Clip stem.
        fps: Frame rate in Hz.
        body_pos: ``(T, 24, 3)`` body origins (SMPL-X joint centres), metres.
        body_quat: ``(T, 24, 4)`` body orientations, MuJoCo ``wxyz``.
    """

    name: str
    fps: float
    body_pos: np.ndarray
    body_quat: np.ndarray

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.body_pos.shape[0])


def load_motion(path: Path) -> YogaReference:
    """Load one ProtoMotions ``.motion`` file.

    Args:
        path: ``.motion`` file.

    Returns:
        The reference, with quaternions converted to ``wxyz``.
    """
    import torch

    data = torch.load(
        path, weights_only=False, map_location="cpu", pickle_module=_STUB_PICKLE
    )
    pos = data["rigid_body_pos"].double().numpy()
    xyzw = data["rigid_body_rot"].double().numpy()
    if pos.shape[1] != len(PLANT_BODIES):
        raise ValueError(
            f"{path}: expected {len(PLANT_BODIES)} bodies, got {pos.shape[1]}"
        )
    wxyz = np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)
    wxyz /= np.linalg.norm(wxyz, axis=-1, keepdims=True)
    return YogaReference(
        name=path.stem, fps=float(data["fps"]), body_pos=pos, body_quat=wxyz
    )


def save_reference(ref: YogaReference, path: Path) -> None:
    """Write a reference as a plain ``.npz`` (no pickled objects)."""
    np.savez_compressed(
        path, name=ref.name, fps=ref.fps, body_pos=ref.body_pos, body_quat=ref.body_quat,
        body_names=np.array(PLANT_BODIES),
    )  # fmt: skip


def load_reference(path: Path) -> YogaReference:
    """Read a reference written by :func:`save_reference`."""
    d = np.load(path, allow_pickle=False)
    if tuple(str(n) for n in d["body_names"]) != PLANT_BODIES:
        raise ValueError(f"{path}: unexpected body order")
    return YogaReference(
        name=str(d["name"]),
        fps=float(d["fps"]),
        body_pos=d["body_pos"],
        body_quat=d["body_quat"],
    )


def base_clip_paths(release_dir: Path) -> list[Path]:
    """Base clips of a release (dwell variants excluded), sorted by name."""
    paths = sorted((release_dir / "motions").glob("*.motion"))
    return [p for p in paths if not p.stem.endswith(_DWELL_SUFFIXES)]


# --------------------------------------------------------------------------- #
# Plant colliders: which plant bodies touch the floor in a reference frame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlantColliders:
    """The plant's one-primitive-per-body colliders, in body frames."""

    body_index: np.ndarray  # (G,) index into PLANT_BODIES
    geom_type: np.ndarray  # (G,) mjtGeom
    pos: np.ndarray  # (G, 3)
    mat: np.ndarray  # (G, 3, 3)
    size: np.ndarray  # (G, 3)
    rest_body_pos: (
        np.ndarray
    )  # (24, 3) body origins in the rest pose (pelvis at origin)


def plant_colliders(plant_xml: Path) -> PlantColliders:
    """Read the plant MJCF's colliders and rest skeleton."""
    model = mujoco.MjModel.from_xml_path(str(plant_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [model.body(b).name for b in range(1, model.nbody)]
    if tuple(names) != PLANT_BODIES:
        raise ValueError(f"{plant_xml}: body order {names} differs from PLANT_BODIES")
    bi, gt, pos, mat, size = [], [], [], [], []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, model.geom_quat[g])
        bi.append(b - 1)
        gt.append(int(model.geom_type[g]))
        pos.append(model.geom_pos[g].copy())
        mat.append(m.reshape(3, 3))
        size.append(model.geom_size[g].copy())
    return PlantColliders(
        body_index=np.array(bi), geom_type=np.array(gt), pos=np.array(pos), mat=np.array(mat),
        size=np.array(size), rest_body_pos=data.xpos[1:].copy(),
    )  # fmt: skip


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` rotation matrices."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1
            ),
            np.stack(
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1
            ),
            np.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def lowest_point_z(
    geom_type: np.ndarray, center: np.ndarray, mat: np.ndarray, size: np.ndarray
) -> np.ndarray:
    """Lowest world z of sphere / capsule / box / ellipsoid primitives.

    Args:
        geom_type: ``(G,)`` mjtGeom values.
        center: ``(..., G, 3)`` world centres.
        mat: ``(..., G, 3, 3)`` world orientations.
        size: ``(G, 3)`` MuJoCo sizes.

    Returns:
        ``(..., G)`` lowest z.
    """
    zrow = mat[..., 2, :]  # world z of each local axis
    out = np.empty(center.shape[:-1])
    for t, sel in (
        (mujoco.mjtGeom.mjGEOM_SPHERE, geom_type == mujoco.mjtGeom.mjGEOM_SPHERE),
        (mujoco.mjtGeom.mjGEOM_CAPSULE, geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE),
        (mujoco.mjtGeom.mjGEOM_BOX, geom_type == mujoco.mjtGeom.mjGEOM_BOX),
        (mujoco.mjtGeom.mjGEOM_ELLIPSOID, geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    ):
        if not sel.any():
            continue
        c, z, s = center[..., sel, 2], zrow[..., sel, :], size[sel]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out[..., sel] = c - s[:, 0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            out[..., sel] = c - np.abs(z[..., 2]) * s[:, 1] - s[:, 0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            out[..., sel] = c - (np.abs(z) * s).sum(-1)
        else:
            out[..., sel] = c - np.sqrt(((z * s) ** 2).sum(-1))
    unsupported = ~np.isin(
        geom_type,
        [
            mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        ],
    )  # fmt: skip
    if unsupported.any():
        raise ValueError(
            f"unsupported collider types {set(geom_type[unsupported].tolist())}"
        )
    return out


def plant_body_lowest(ref: YogaReference, colliders: PlantColliders) -> np.ndarray:
    """``(T, 24)`` lowest collider point of every plant body (inf for bodies without one)."""
    rot = quat_to_mat(ref.body_quat)  # (T, 24, 3, 3)
    rb = rot[:, colliders.body_index]  # (T, G, 3, 3)
    center = ref.body_pos[:, colliders.body_index] + np.einsum(
        "tgij,gj->tgi", rb, colliders.pos
    )
    gmat = np.einsum("tgij,gjk->tgik", rb, colliders.mat)
    low = lowest_point_z(colliders.geom_type, center, gmat, colliders.size)
    out = np.full((ref.n_frames, len(PLANT_BODIES)), np.inf)
    for g, b in enumerate(colliders.body_index):
        out[:, b] = np.minimum(out[:, b], low[:, g])
    return out


def main() -> None:
    """Export the base clips of a release to ``.npz`` references."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for p in base_clip_paths(args.release):
        ref = load_motion(p)
        save_reference(ref, args.out / f"{ref.name}.npz")
        print(f"{ref.name}: {ref.n_frames} frames @ {ref.fps:g} Hz")


if __name__ == "__main__":
    main()
