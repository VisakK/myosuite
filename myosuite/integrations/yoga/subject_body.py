# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoFullBody scaled to one performer's skeleton and masses.

The yoga references live on the performer's own skeleton (SMPL-X joint centres of
her personal template, "plant v2"). This module measures that skeleton against
MyoFullBody and edits the MyoFullBody ``MjSpec`` the way OpenSim's Scale tool
edits a model:

* every body gets a linear map ``A_b`` (in its own frame) applied to everything it
  owns that is a *point*: child-body offsets, sites (muscle path points), geom
  centres and ``fromto`` ends, wrap objects and the centre of mass;
* sizes (colliders, wrap cylinders/spheres, bone meshes) scale with the body's
  isotropic factor;
* coupled translations (knee, patella polynomials) scale with the parent's factor;
* body masses move to her per-segment masses (template volume), inertias follow
  ``m r^2``;
* muscle ``lengthrange`` is multiplied by the measured musculotendon-length ratio
  of each muscle, so the optimal fibre length and tendon slack length scale with
  the path (the MyoConverter calibration is kept in normalised units).

An optional 3-DOF neck (MyoFullBody has none) with three torque motors is added
for poses that bend the neck (shoulderstand, plow).

The result is stored as a JSON "subject table"; :func:`build_subject_spec`
re-applies it to a fresh MyoFullBody spec in milliseconds, so the same model is
built by the CPU env, the mjlab task and the retargeter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.yoga.reference_io import PLANT_BODIES

SIDES = ("l", "r")
PLANT_SIDE = {"l": "L", "r": "R"}

# Myo world axes at qpos0: x = left, -y = forward, z = up.
MYO_LATERAL = np.array([1.0, 0.0, 0.0])
MYO_FORWARD = np.array([0.0, -1.0, 0.0])
MYO_UP = np.array([0.0, 0.0, 1.0])

NECK_JOINTS = ("neck_flexion", "neck_lateral", "neck_rotation")
NECK_ACTUATORS = tuple(f"{j}_motor" for j in NECK_JOINTS)

# Joint ranges (deg) for her yoga corpus: MyoFullBody's range joined with the p0.5-p99.5
# angles her 56 clips need when retargeted with an unconstraining box (the range audit in
# README.md); twist and foot-roll DOFs are capped where the audit hit its own box (slack,
# not need). Signs: flex_extension + = extension, hip/knee + = flexion, ankle + =
# dorsiflexion, mtp - = toe extension.
YOGA_RANGES_DEG: dict[str, tuple[float, float]] = {
    "flex_extension": (-77.3, 58.0), "lat_bending": (-28.0, 33.0), "axial_rotation": (-70.0, 62.0),
    **{f"elv_angle_{s}": (-125.0, 138.0) for s in SIDES},
    **{f"shoulder_rot_{s}": (-105.0, 135.0) for s in SIDES},
    **{f"elbow_flex_{s}": (0.0, 145.0) for s in SIDES},
    **{f"pro_sup_{s}": (-100.0, 100.0) for s in SIDES},
    **{f"deviation_{s}": (-25.0, 35.0) for s in SIDES},
    **{f"flexion_{s}": (-95.0, 88.0) for s in SIDES},
    **{f"hip_flexion_{s}": (-55.0, 155.0) for s in SIDES},
    **{f"hip_adduction_{s}": (-75.0, 32.0) for s in SIDES},
    **{f"hip_rotation_{s}": (-70.0, 75.0) for s in SIDES},
    **{f"knee_angle_{s}": (0.0, 160.0) for s in SIDES},
    **{f"ankle_angle_{s}": (-70.0, 55.0) for s in SIDES},
    **{f"subtalar_angle_{s}": (-30.0, 30.0) for s in SIDES},
    **{f"mtp_angle_{s}": (-80.0, 45.0) for s in SIDES},
}  # fmt: skip
# The knee couplings (tibia/patella translations and rotations) were fitted on 0-120 deg.
KNEE_COUPLING_FIT_DEG = 120.0

# Extrinsic finger flexors kept by ``SubjectTable.finger_flexors`` (right names; left: "_left").
FINGER_FLEXORS: tuple[str, ...] = (
    *(f"FDS{i}" for i in range(2, 6)), *(f"FDP{i}" for i in range(2, 6)), "FPL",
)  # fmt: skip
# Wrist obstacles (radius m, side +1 palmar / -1 dorsal). The radii are measured flexion
# moment arms, roughly constant over the range: FCR, FCU, ECRL, ECRB, ECU from Garland et
# al. 2018 (12.2, 11.3, 6.6, 10.8, 4.0 mm); FDS/FDP ~1.1 cm at 40-60 deg extension and
# FPL from Gonzalez et al. 1997; PL below its 2.1 cm at neutral.
WRIST_OBSTACLES: dict[str, tuple[float, int]] = {
    "FCR": (0.012, 1), "FCU": (0.011, 1), "PL": (0.014, 1),
    "ECRL": (0.007, -1), "ECRB": (0.010, -1), "ECU": (0.004, -1),
    **{f"FDS{i}": (0.011, 1) for i in range(2, 6)},
    **{f"FDP{i}": (0.011, 1) for i in range(2, 6)},
    "FPL": (0.010, 1),
}  # fmt: skip
# Wrap spheres up to this radius in a wrist path act as via points (see _repair_wrist_paths).
WRIST_TINY_WRAP_M = 0.003
WRAP_GROUP = 3
# A stiff end-range: the limit's force per radian is ~ d / (1 - d) * k * m_eff. Measured on the
# seated handstand wrist: ~9 N m per degree past the limit, ~6x MuJoCo's default (solref 0.02,
# solimp 0.9-0.95). The time constant stays above refsafe's 2 * timestep.
STIFF_LIMIT_SOLREF: tuple[float, ...] = (0.01, 1.0)
STIFF_LIMIT_SOLIMP: tuple[float, ...] = (0.99, 0.995, 0.001, 0.5, 2.0)
# Active-F0 factors (right names; applied to both sides). SUPSP's gainprm holds the
# supinator's 339.1 N; its biasprm and the older MyoArm keep 496.7 N (MoBL-ARMS 499.2 N).
SUPSP_FIX: dict[str, float] = {"SUPSP": 496.738 / 339.132}
# TMIN, TMAJ, INFSP and LAT2 are 2.0, 2.0, 2.0 and 1.69x their MoBL-ARMS forces (269.5,
# 144.0, 1075.8, 317.5 N), an undocumented edit; these factors restore the MoBL-ARMS values.
SHOULDER_EDIT_UNDO: dict[str, float] = {
    "TMIN": 269.5 / 539.0, "TMAJ": 144.0 / 288.0, "INFSP": 1075.8 / 2150.71, "LAT2": 317.5 / 535.42,
}  # fmt: skip
# Wrist extension (deg) the round-3 plant allows: her clips need at least 100 deg (the range
# audit's box), and a flat hand on the floor needs the hand seated past MyoArm's 45 deg.
R3_WRIST_EXTENSION_DEG = 110.0

# Plant body groups whose masses define each MyoFullBody mass group.
PLANT_MASS_GROUPS: dict[str, tuple[str, ...]] = {
    "trunk": ("Pelvis", "Torso", "Spine", "Chest", "L_Thorax", "R_Thorax"),
    "head": ("Neck", "Head"),
    **{f"upperarm_{s}": (f"{PLANT_SIDE[s]}_Shoulder",) for s in SIDES},
    **{f"forearm_{s}": (f"{PLANT_SIDE[s]}_Elbow",) for s in SIDES},
    **{f"hand_{s}": (f"{PLANT_SIDE[s]}_Wrist", f"{PLANT_SIDE[s]}_Hand") for s in SIDES},
    **{f"thigh_{s}": (f"{PLANT_SIDE[s]}_Hip",) for s in SIDES},
    **{f"shank_{s}": (f"{PLANT_SIDE[s]}_Knee",) for s in SIDES},
    **{f"foot_{s}": (f"{PLANT_SIDE[s]}_Ankle", f"{PLANT_SIDE[s]}_Toe") for s in SIDES},
}


def _descendants(model: mujoco.MjModel, root: str) -> list[str]:
    rid = model.body(root).id
    out = []
    for b in range(model.nbody):
        p = b
        while p not in (0, rid):
            p = int(model.body_parentid[p])
        if p == rid:
            out.append(model.body(b).name)
    return out


def myo_body_groups(model: mujoco.MjModel) -> dict[str, str]:
    """Body name -> segment group (scale and mass group) for MyoFullBody."""
    groups: dict[str, str] = {}
    for b in range(1, model.nbody):
        groups[model.body(b).name] = "trunk"
    groups["neck"] = groups["head"] = "head"
    for s in SIDES:
        for n in _descendants(model, f"humerus_{s}"):
            groups[n] = f"upperarm_{s}"
        for n in _descendants(model, f"ulna_{s}"):
            groups[n] = f"forearm_{s}"
        for n in _descendants(model, f"lunate_{s}"):
            groups[n] = f"hand_{s}"
        for n in _descendants(model, f"femur_{s}"):
            groups[n] = f"thigh_{s}"
        for n in _descendants(model, f"tibia_{s}"):
            groups[n] = f"shank_{s}"
        for n in _descendants(model, f"calcn_{s}"):
            groups[n] = f"foot_{s}"
    return groups


# --------------------------------------------------------------------------- #
# Rest-pose landmarks of both skeletons
# --------------------------------------------------------------------------- #
def myo_joint_centres(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, np.ndarray]:
    """World joint centres of MyoFullBody (``data`` must be forwarded)."""
    xp = lambda n: data.xpos[model.body(n).id].copy()  # noqa: E731
    lm: dict[str, np.ndarray] = {}
    for s in SIDES:
        lm[f"hip_{s}"] = xp(f"femur_{s}")
        lm[f"knee_{s}"] = xp(f"tibia_{s}")
        lm[f"ankle_{s}"] = xp(f"talus_{s}")
        lm[f"mtp_{s}"] = xp(f"toes_{s}")
        lm[f"shoulder_{s}"] = xp(f"humerus_{s}")
        lm[f"elbow_{s}"] = xp(f"ulna_{s}")
        lm[f"wrist_{s}"] = xp(f"lunate_{s}")
        lm[f"hand_{s}"] = np.mean([xp(f"{k}proxph_{s}") for k in (2, 3, 4, 5)], axis=0)
    return lm


def plant_joint_centres(rest_body_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Plant (her SMPL-X) joint centres from the rest-pose body origins."""
    p = {n: rest_body_pos[i] for i, n in enumerate(PLANT_BODIES)}
    lm: dict[str, np.ndarray] = {}
    for s in SIDES:
        S = PLANT_SIDE[s]
        lm[f"hip_{s}"] = p[f"{S}_Hip"]
        lm[f"knee_{s}"] = p[f"{S}_Knee"]
        lm[f"ankle_{s}"] = p[f"{S}_Ankle"]
        lm[f"mtp_{s}"] = p[f"{S}_Toe"]
        lm[f"shoulder_{s}"] = p[f"{S}_Shoulder"]
        lm[f"elbow_{s}"] = p[f"{S}_Elbow"]
        lm[f"wrist_{s}"] = p[f"{S}_Wrist"]
        lm[f"hand_{s}"] = p[f"{S}_Hand"]
    return lm


def segment_lengths(lm: dict[str, np.ndarray]) -> dict[str, float]:
    """Pose-invariant segment lengths between joint centres."""
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))  # noqa: E731
    out = {
        "hip_width": d("hip_l", "hip_r"),
        "shoulder_width": d("shoulder_l", "shoulder_r"),
    }
    for s in SIDES:
        out[f"thigh_{s}"] = d(f"hip_{s}", f"knee_{s}")
        out[f"shank_{s}"] = d(f"knee_{s}", f"ankle_{s}")
        out[f"foot_{s}"] = d(f"ankle_{s}", f"mtp_{s}")
        out[f"upperarm_{s}"] = d(f"shoulder_{s}", f"elbow_{s}")
        out[f"forearm_{s}"] = d(f"elbow_{s}", f"wrist_{s}")
        out[f"hand_{s}"] = d(f"wrist_{s}", f"hand_{s}")
    return out


def collider_extent(
    model: mujoco.MjModel, data: mujoco.MjData, bodies: list[str], direction: np.ndarray
) -> float:
    """Largest world projection on *direction* over the colliders of *bodies* (support function)."""
    best = -np.inf
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    ids = {model.body(b).id for b in bodies}
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in ids or not model.geom_contype[g]:
            continue
        c, R, s, t = (
            data.geom_xpos[g],
            data.geom_xmat[g].reshape(3, 3),
            model.geom_size[g],
            model.geom_type[g],
        )
        ul = R.T @ u
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            h = s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            h = abs(ul[2]) * s[1] + s[0]
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            h = float(np.linalg.norm(s * ul))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            h = float(np.abs(ul) @ s)
        else:
            continue
        best = max(best, float(c @ u) + h)
    return best


# SMPL-X vertex ids of surface landmarks (smplx ``vertex_ids['smplx']``).
SMPLX_VERTICES = {
    "heel_l": 8846, "bigtoe_l": 5770, "heel_r": 8635, "bigtoe_r": 8463,
    "middletip_l": 5058, "middletip_r": 7794, "nose": 9120,
}  # fmt: skip
# Plant body frame (x forward, y left, z up) from SMPL-X canonical axes: ``v_plant = AXES.T @ v_smplx``.
SMPLX_TO_PLANT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def her_template(mosh_pkl: Path, smplx_npz: Path) -> dict[str, np.ndarray]:
    """Her rest-pose joints and surface landmarks in the plant frame (pelvis at origin).

    Args:
        mosh_pkl: Any MoSh++ stage-II fit of the performer (all share her template).
        smplx_npz: SMPL-X model file of the fit's gender (``SMPLX_FEMALE.npz``).

    Returns:
        ``joints`` (55, 3), each landmark of :data:`SMPLX_VERTICES`, ``crown`` and ``sole``.
    """
    import pickle

    with open(mosh_pkl, "rb") as f:
        fit = pickle.load(f, encoding="latin1")
    vt = np.asarray(fit["stagei_debug_details"]["v_template"], float)
    J = np.load(smplx_npz, allow_pickle=True)["J_regressor"] @ vt
    to_plant = lambda p: (p - J[0]) @ SMPLX_TO_PLANT  # noqa: E731
    V = to_plant(vt)
    out = {
        "joints": to_plant(J),
        "crown": V[V[:, 2].argmax()],
        "sole": V[V[:, 2].argmin()],
    }
    out.update({k: V[v] for k, v in SMPLX_VERTICES.items()})
    return out


# --------------------------------------------------------------------------- #
# Subject table
# --------------------------------------------------------------------------- #
@dataclass
class SubjectTable:
    """Everything needed to turn MyoFullBody into the subject's body.

    Attributes:
        segment_scale: Isotropic factor per scale group (``thigh_l`` ...).
        hip_width_scale: Lateral factor of the pelvis/sacrum (hip joint centres).
        clavicle_scale: Isotropic factor of each clavicle (sets shoulder width).
        head_lift: Metres added to the head's height (crown match).
        group_mass: Target mass (kg) of each mass group.
        lengthrange: Muscle name -> calibrated ``lengthrange``.
        neck: Whether to add the torque-driven 3-DOF neck.
        neck_gear: Neck motor gear (N m at ctrl 1).
        joint_ranges: Joint name -> ``[lo, hi]`` (rad) replacing MyoFullBody's range
            (her ranges exceed the model's normative ones, see ``README.md``).
        coupling_extension: Independent joint name -> the upper bound (rad) up to which
            its coupling polynomials were fitted; beyond it every polynomial driven by
            that joint is continued linearly (C1) instead of extrapolated.
        self_collision: Let the transplanted primitives collide with each other (as the
            plant did: every pair except anatomically adjacent segments) instead of
            MyoFullBody's 75 explicit pairs on its own geometry. Braces such as crow's
            shins on the upper arms carry load; without this they pass through.
        passive_scale: Muscle name -> factor on ``fpmax`` (passive force only; the
            active force-length curve is unchanged). See
            :func:`myosuite.integrations.yoga.statics.fit_passive_compliance`.
        finger_flexors: Keep the extrinsic finger flexors (:data:`FINGER_FLEXORS`), which
            MuscleMimic removes with the finger joints. The fingers stay welded straight,
            so these muscles act on the wrist (and the elbow).
        wrist_obstacles: Muscle -> ``[radius (m), side]`` (side +1 palmar, -1 dorsal):
            a cylinder on the wrist flexion axis that the muscle's path wraps around
            whenever it would pass closer to the axis (:func:`_repair_wrist_paths`).
            MyoArm's paths are built for ±45° of wrist flexion and lose or invert their
            moment arms beyond it.
        limit_solref: Joint name -> ``solref`` of its range limit.
        limit_solimp: Joint name -> ``solimp`` of its range limit. A stiff end-range
            (ligament and bone, :data:`STIFF_LIMIT_SOLREF`) instead of MuJoCo's default
            soft limit, whose stiffness scales with a light distal segment's inertia.
        strength_scale: Muscle name -> factor on the active peak force F0
            (``gainprm[2]``); the passive force (``biasprm``) keeps its fitted value.
        colliders: Floor-contact primitives replacing MyoFullBody's own colliders
            (which stay as visual geoms): ``{body, type, size, pos, quat}`` in the
            MyoFullBody body frame. Filled by
            :func:`myosuite.integrations.yoga.retarget.transplant_plant_colliders` with the
            plant's primitives, so the muscle model touches the floor exactly where the
            PhysX teacher's body did.
        measured: Audit numbers (segment lengths of both skeletons, masses).
    """

    segment_scale: dict[str, float]
    hip_width_scale: float
    clavicle_scale: dict[str, float]
    head_lift: float
    group_mass: dict[str, float]
    lengthrange: dict[str, list[float]] = field(default_factory=dict)
    neck: bool = True
    neck_gear: float = 30.0
    joint_ranges: dict[str, list[float]] = field(default_factory=dict)
    coupling_extension: dict[str, float] = field(default_factory=dict)
    colliders: list[dict] = field(default_factory=list)
    self_collision: bool = False
    passive_scale: dict[str, float] = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    finger_flexors: bool = False
    wrist_obstacles: dict[str, list[float]] = field(default_factory=dict)
    limit_solref: dict[str, list[float]] = field(default_factory=dict)
    limit_solimp: dict[str, list[float]] = field(default_factory=dict)
    strength_scale: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: Path) -> SubjectTable:
        """Read a JSON table."""
        return SubjectTable(**json.loads(Path(path).read_text()))


def base_fullbody_spec(finger_flexors: bool = False) -> mujoco.MjSpec:
    """A private copy of the MuscleMimic MyoFullBody spec (fingers removed, mimic sites).

    Args:
        finger_flexors: Keep the extrinsic finger flexors (:data:`FINGER_FLEXORS`). The
            finger joints are removed either way, so the fingers stay welded straight.
    """
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    cfg = dict(default_mimic_fullbody_config())
    cfg["disable_fingers"] = not finger_flexors
    spec, _ = build_mimic_fullbody_spec(config_dict.create(**cfg))
    if finger_flexors:
        _weld_fingers(spec, keep=FINGER_FLEXORS)
    return spec


def _weld_fingers(spec: mujoco.MjSpec, keep: tuple[str, ...]) -> None:
    """Remove the finger joints and every finger muscle except *keep* (both sides)."""
    from myosuite.integrations.musclemimic.bimanual_model import (
        FINGER_JOINT_TOKENS,
        FINGER_MUSCLE_TOKENS,
    )

    kept = {n for k in keep for n in (k, f"{k}_left")}
    for joint in [j for j in spec.joints if j.name in FINGER_JOINT_TOKENS]:
        spec.delete(joint)
    gone = [
        a
        for a in spec.actuators
        if a.name in FINGER_MUSCLE_TOKENS and a.name not in kept
    ]
    tendons = {a.target for a in gone}
    for act in gone:
        spec.delete(act)
    for tendon in [t for t in spec.tendons if t.name in tendons]:
        spec.delete(tendon)


def compile_spec(spec: mujoco.MjSpec, timestep: float = 0.002) -> mujoco.MjModel:
    """Compile with the MuscleMimic CPU options (Euler, ``sim_dt``)."""
    model = spec.compile()
    model.opt.timestep = timestep
    return model


def _rest(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def _body_rotation(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    return data.xmat[model.body(name).id].reshape(3, 3)


def _linear_maps(
    model: mujoco.MjModel, table: SubjectTable
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per-body point map ``A_b`` (body frame) and isotropic size factor."""
    data = _rest(model)
    groups = myo_body_groups(model)
    A: dict[str, np.ndarray] = {}
    s: dict[str, float] = {}
    for b in range(1, model.nbody):
        name = model.body(b).name
        k = float(table.segment_scale.get(groups[name], 1.0))
        A[name], s[name] = k * np.eye(3), k
    for name in ("pelvis", "sacrum"):
        R = _body_rotation(data, model, name)
        Sw = np.eye(3) + (table.hip_width_scale - 1.0) * np.outer(
            MYO_LATERAL, MYO_LATERAL
        )
        A[name] = R.T @ Sw @ R
    for side in SIDES:
        A[f"clavicle_{side}"] = table.clavicle_scale[side] * np.eye(3)
        s[f"clavicle_{side}"] = table.clavicle_scale[side]
    return A, s


def _scale_geometry(
    spec: mujoco.MjSpec, model: mujoco.MjModel, table: SubjectTable
) -> None:
    """Apply the point maps, size factors and coupling scales to *spec* in place."""
    A, s = _linear_maps(model, table)
    data = _rest(model)
    meshes = {m.name: m for m in spec.meshes}
    for body in spec.bodies:
        if body.name not in A:
            continue
        Ab, sb = A[body.name], s[body.name]
        for child in body.bodies:
            child.pos = Ab @ np.asarray(child.pos)
        for site in body.sites:
            site.pos = Ab @ np.asarray(site.pos)
            site.size = np.asarray(site.size) * sb
        for geom in body.geoms:
            if not np.isnan(geom.fromto[0]):
                ft = np.asarray(geom.fromto)
                geom.fromto = np.concatenate([Ab @ ft[:3], Ab @ ft[3:]])
                geom.size = np.asarray(geom.size) * np.array([sb, 1.0, 1.0])
                continue
            geom.pos = Ab @ np.asarray(geom.pos)
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                if sb != 1.0:
                    meshes[geom.meshname].scale = (
                        np.asarray(meshes[geom.meshname].scale) * sb
                    )
            elif geom.type != mujoco.mjtGeom.mjGEOM_PLANE:
                geom.size = np.asarray(geom.size) * sb
        body.ipos = Ab @ np.asarray(body.ipos)
    head = spec.body("head")
    R_neck = _body_rotation(data, model, "neck")
    head.pos = np.asarray(head.pos) + R_neck.T @ (table.head_lift * MYO_UP)
    # Coupled translations of a child scale with its parent segment (OpenSim CustomJoint rule).
    slide_scale: dict[str, float] = {}
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            body = int(model.jnt_bodyid[model.joint(joint.name).id])
            k = s.get(model.body(int(model.body_parentid[body])).name, 1.0)
            slide_scale[joint.name] = k
            joint.range = np.asarray(joint.range) * k
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT and eq.name1 in slide_scale:
            d = np.asarray(eq.data).copy()
            d[:5] *= slide_scale[eq.name1]
            eq.data = d


def _apply_ranges(
    spec: mujoco.MjSpec, model: mujoco.MjModel, table: SubjectTable
) -> None:
    """Range overrides, C1 continuation of fitted couplings, consistent dependent ranges.

    A coupling polynomial was fitted on the independent joint's original range;
    beyond ``coupling_extension[joint]`` it is refitted (degree 4, least squares) to the
    original polynomial inside and its tangent line outside. Every dependent joint's
    range is then widened to cover its coupling's values over the independent joint's
    (new) range, so joint limits never contradict an equality.
    """
    joints = {j.name: j for j in spec.joints}
    for name, rng in table.joint_ranges.items():
        joints[name].range = np.asarray(rng, float)
    new_range = {j.name: np.asarray(j.range, float) for j in spec.joints}
    for eq in spec.equalities:
        if eq.type != mujoco.mjtEq.mjEQ_JOINT or not eq.name2:
            continue
        c = np.asarray(eq.data, float).copy()
        x0 = float(model.qpos0[model.jnt_qposadr[model.joint(eq.name2).id]])
        lo, hi = new_range[eq.name2]
        poly = lambda x, k=c[:5]: np.polyval(k[::-1], x - x0)  # noqa: E731
        if eq.name2 in table.coupling_extension:
            fit_hi = float(table.coupling_extension[eq.name2])
            xs = np.linspace(lo, hi, 400)
            slope = float(np.polyval(np.polyder(c[:5][::-1]), fit_hi - x0))
            ys = np.where(xs <= fit_hi, poly(xs), poly(fit_hi) + slope * (xs - fit_hi))
            c[:5] = np.polyfit(xs - x0, ys, 4)[::-1]
            eq.data = c
        y0 = float(model.qpos0[model.jnt_qposadr[model.joint(eq.name1).id]])
        vals = y0 + np.polyval(c[:5][::-1], np.linspace(lo, hi, 400) - x0)
        dep = joints[eq.name1]
        r = np.asarray(dep.range, float)
        margin = 0.02 * max(r[1] - r[0], 1e-3)
        dep.range = np.array(
            [min(r[0], vals.min() - margin), max(r[1], vals.max() + margin)]
        )


def _freeze_inertials(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    """Give every body an explicit inertial equal to its compiled one.

    Head, neck, carpals, talus, toes and patella take their mass from geom densities;
    without this, scaling their meshes would change their mass by ``s^3`` and the
    group mass targets could not be met.
    """
    for body in spec.bodies:
        if not body.name or body.explicitinertial:
            continue
        b = model.body(body.name).id
        body.explicitinertial = 1
        body.mass = float(model.body_mass[b])
        body.ipos = model.body_ipos[b].copy()
        body.iquat = model.body_iquat[b].copy()
        body.inertia = model.body_inertia[b].copy()
        body.fullinertia = np.full(6, np.nan)


def _scale_masses(
    spec: mujoco.MjSpec, model: mujoco.MjModel, table: SubjectTable
) -> None:
    groups = myo_body_groups(model)
    _, s = _linear_maps(model, table)
    base_mass: dict[str, float] = {}
    for b in range(1, model.nbody):
        g = groups[model.body(b).name]
        base_mass[g] = base_mass.get(g, 0.0) + float(model.body_mass[b])
    for body in spec.bodies:
        if body.name not in groups:
            continue
        ratio = table.group_mass[groups[body.name]] / base_mass[groups[body.name]]
        body.mass = float(body.mass) * ratio
        k2 = ratio * s[body.name] ** 2
        if not np.isnan(body.fullinertia[0]):
            body.fullinertia = np.asarray(body.fullinertia) * k2
        else:
            body.inertia = np.asarray(body.inertia) * k2


def _add_neck(spec: mujoco.MjSpec, model: mujoco.MjModel, gear: float) -> None:
    """Three hinges at the cervical base (3 cm above the shoulders) + torque motors."""
    data = _rest(model)
    gh = 0.5 * (
        data.xpos[model.body("humerus_l").id] + data.xpos[model.body("humerus_r").id]
    )
    pivot_w = gh + 0.03 * MYO_UP + 0.009 * MYO_FORWARD
    hid = model.body("head").id
    R = data.xmat[hid].reshape(3, 3)
    pivot = R.T @ (pivot_w - data.xpos[hid])
    head = spec.body("head")
    ranges = {"neck_flexion": 1.2, "neck_lateral": 0.7, "neck_rotation": 1.2}
    for name, axis_w in zip(NECK_JOINTS, (MYO_LATERAL, MYO_FORWARD, MYO_UP)):
        head.add_joint(
            name=name, type=mujoco.mjtJoint.mjJNT_HINGE, pos=pivot, axis=R.T @ axis_w,
            range=[-ranges[name], ranges[name]], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=2.0, damping=0.5, armature=0.005,
        )  # fmt: skip
        act = spec.add_actuator(
            name=f"{name}_motor", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        act.set_to_motor()
        act.gear = [gear, 0, 0, 0, 0, 0]
        act.ctrlrange = [-1.0, 1.0]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE


_TENDON_ATTRS = (
    "width", "rgba", "group", "material", "limited", "range", "actfrclimited", "actfrcrange",
    "stiffness", "damping", "springlength", "margin", "frictionloss", "armature",
    "solref_limit", "solimp_limit", "solref_friction", "solimp_friction", "userdata",
)  # fmt: skip


def _arm_name(name: str, side: str) -> str:
    """MyoArm names carry no suffix on the right and ``_left`` on the left."""
    return name if side == "r" else f"{name}_left"


def _tendon_path(model: mujoco.MjModel, tendon: str) -> list[tuple[str, ...]]:
    """``("site", name)`` and ``("geom", name, sidesite)`` items of a spatial tendon."""
    tid = model.tendon(tendon).id
    out: list[tuple[str, ...]] = []
    for w in range(
        model.tendon_adr[tid], model.tendon_adr[tid] + model.tendon_num[tid]
    ):
        obj = int(model.wrap_objid[w])
        if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE:
            out.append(("site", model.site(obj).name))
        else:
            side = int(model.wrap_prm[w])
            out.append(
                (
                    "geom",
                    model.geom(obj).name,
                    model.site(side).name if side >= 0 else "",
                )
            )
    return out


def _rebuild_tendon(
    spec: mujoco.MjSpec, name: str, path: list[tuple[str, ...]]
) -> None:
    """Replace a spatial tendon by one with the same name, attributes and a new path.

    (``MjsTendonPath`` cannot be edited in place; actuators refer to tendons by name.)
    """
    old = spec.tendon(name)
    attrs = {a: getattr(old, a) for a in _TENDON_ATTRS}
    spec.delete(old)
    new = spec.add_tendon(name=name)
    for a, v in attrs.items():
        setattr(new, a, v.copy() if isinstance(v, np.ndarray) else v)
    for item in path:
        if item[0] == "site":
            new.wrap_site(item[1])
        else:
            new.wrap_geom(item[1], item[2])


def _palmar_direction(model: mujoco.MjModel, side: str) -> np.ndarray:
    """Unit palmar direction in the lunate frame, perpendicular to the wrist flexion axis."""
    data = _rest(model)
    lb = model.body(f"lunate_{side}").id
    R, p = data.xmat[lb].reshape(3, 3), data.xpos[lb]
    axis = model.jnt_axis[model.joint(f"flexion_{side}").id]
    hand = R.T @ (data.xpos[model.body(f"thirdmc_{side}").id] - p)
    hand = hand - axis * (axis @ hand)
    palmar = np.cross(axis, hand / np.linalg.norm(hand))
    flexor = R.T @ (data.site_xpos[model.site(_arm_name("FCR-P2", side)).id] - p)
    return palmar if palmar @ flexor > 0 else -palmar


def _repair_wrist_paths(
    spec: mujoco.MjSpec, model: mujoco.MjModel, table: SubjectTable
) -> None:
    """Keep the wrist muscles' moment arms physiological beyond MyoArm's ±45° range.

    MyoArm's wrist paths come from a model built for ±45° of flexion. Further out the
    flexors' moment arms fall to 0.3-0.5 cm (measured: ~1.1-1.2 cm, roughly constant) and
    ECRL/ECRB turn into flexors, because the insertions sit within a centimetre of the
    axis and sweep across it. For each muscle of ``table.wrist_obstacles``, a cylinder on
    the flexion axis is put in the path segment that crosses the wrist (last point on the
    forearm to first point on the hand). The tendon wraps around it on its anatomical
    side whenever it would pass closer to the axis than the radius; inside ±45° the paths
    stay outside it. Wrap spheres of at most :data:`WRIST_TINY_WRAP_M` become via points
    at their centres, except on the dorsal side of the hand, where a point fixed to the
    hand passes over the joint beyond ~70° of extension and inverts the extensor: there
    they are dropped.

    Args:
        spec: Spec being built (geometry already scaled).
        model: The compiled base model (path layout, rest geometry).
        table: Subject table (obstacles, hand scale).
    """
    muscles = {model.actuator(i).name: i for i in range(model.nu)}
    quat = np.zeros(4)
    for side in SIDES:
        hand = {model.body(n).id for n in _descendants(model, f"lunate_{side}")}
        jid = model.joint(f"flexion_{side}").id
        centre, axis = model.jnt_pos[jid], model.jnt_axis[jid]
        mujoco.mju_quatZ2Vec(quat, axis)
        palmar = _palmar_direction(model, side)
        lunate = spec.body(f"lunate_{side}")
        k = float(table.segment_scale.get(f"hand_{side}", 1.0))
        for base_name, (radius, sgn) in table.wrist_obstacles.items():
            muscle = _arm_name(base_name, side)
            if muscle not in muscles:
                continue
            tendon = model.tendon(int(model.actuator_trnid[muscles[muscle], 0])).name
            r = radius * k
            geom = lunate.add_geom(
                name=f"{muscle}_wrist_obstacle", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r, 0.025, 0.0], pos=centre, quat=quat, contype=0, conaffinity=0,
                group=WRAP_GROUP, rgba=[0.19, 0.83, 0.78, 0.2],
            )  # fmt: skip
            sidesite = lunate.add_site(
                name=f"{muscle}_wrist_side", pos=centre + sgn * palmar * (r + 0.004),
                size=[0.001, 0.0, 0.0], group=WRAP_GROUP,
            )  # fmt: skip
            path: list[tuple] = []
            for item in _tendon_path(model, tendon):
                if item[0] == "site":
                    path.append((*item, int(model.site_bodyid[model.site(item[1]).id])))
                    continue
                gid = model.geom(item[1]).id
                body = int(model.geom_bodyid[gid])
                if model.geom_size[gid][0] > WRIST_TINY_WRAP_M:
                    path.append(item)
                elif sgn > 0 or body not in hand:
                    via = spec.body(model.body(body).name).add_site(
                        name=f"{muscle}_{item[1]}_via", pos=np.asarray(spec.geom(item[1]).pos),
                        size=[0.001, 0.0, 0.0], group=WRAP_GROUP,
                    )  # fmt: skip
                    path.append(("site", via.name, body))
            sites = [i for i, it in enumerate(path) if it[0] == "site"]
            j = next(
                j for j in range(1, len(sites))
                if path[sites[j]][2] in hand and path[sites[j - 1]][2] not in hand
            )  # fmt: skip
            path = (
                path[: sites[j - 1] + 1]
                + [("geom", geom.name, sidesite.name)]
                + path[sites[j] :]
            )
            _rebuild_tendon(
                spec, tendon, [it[:2] if it[0] == "site" else it for it in path]
            )


def build_subject_spec(
    table: SubjectTable, base: mujoco.MjSpec | None = None
) -> mujoco.MjSpec:
    """MyoFullBody spec edited to the subject (geometry, masses, muscles, neck).

    Args:
        table: Subject table (see :func:`fit_subject_table`).
        base: Spec to edit; a fresh MuscleMimic full-body spec by default.

    Returns:
        The edited (uncompiled) spec.
    """
    spec = base_fullbody_spec(table.finger_flexors) if base is None else base
    model = compile_spec(spec.copy())
    _freeze_inertials(spec, model)
    _scale_geometry(spec, model, table)
    _scale_masses(spec, model, table)
    _apply_ranges(spec, model, table)
    if table.wrist_obstacles:
        _repair_wrist_paths(spec, model, table)
    for act in spec.actuators:
        if act.name in table.lengthrange:
            act.lengthrange = np.asarray(table.lengthrange[act.name], float)
        if act.name in table.passive_scale:
            for prm in ("gainprm", "biasprm"):
                v = np.asarray(getattr(act, prm), float).copy()
                v[7] *= table.passive_scale[act.name]  # fpmax
                setattr(act, prm, v)
        if act.name in table.strength_scale:
            v = np.asarray(act.gainprm, float).copy()
            v[2] *= table.strength_scale[act.name]  # active F0 only
            act.gainprm = v
    for joint in spec.joints:
        if joint.name in table.limit_solref:
            joint.solref_limit = np.asarray(table.limit_solref[joint.name], float)
        if joint.name in table.limit_solimp:
            joint.solimp_limit = np.asarray(table.limit_solimp[joint.name], float)
    if table.colliders:
        _replace_colliders(spec, table.colliders, table.self_collision)
    if table.neck:
        _add_neck(spec, model, table.neck_gear)
    return spec


COLLIDER_GROUP = 3  # hidden in default renders


# Adjacent segments of the plant (parent-child) as MyoFullBody host bodies: never collide.
ADJACENT_HOSTS: tuple[tuple[str, str], ...] = (
    ("pelvis", "lumbar3"), ("lumbar3", "lumbar1"), ("lumbar1", "torso"), ("torso", "head"),
    *((a.format(s=s), b.format(s=s)) for s in SIDES for a, b in (
        ("pelvis", "femur_{s}"), ("femur_{s}", "tibia_{s}"), ("tibia_{s}", "calcn_{s}"),
        ("calcn_{s}", "toes_{s}"), ("torso", "clavicle_{s}"), ("clavicle_{s}", "humerus_{s}"),
        ("humerus_{s}", "ulna_{s}"), ("ulna_{s}", "lunate_{s}"))),
)  # fmt: skip


def _replace_colliders(
    spec: mujoco.MjSpec, colliders: list[dict], self_collision: bool = False
) -> None:
    """Turn every body collider into a visual geom and add the given primitives.

    With *self_collision* the primitives also collide with each other, except on
    :data:`ADJACENT_HOSTS` (and within one body).
    """
    for geom in spec.geoms:
        if (
            geom.parent.name
            and geom.parent.name != "world"
            and (geom.contype or geom.conaffinity)
        ):
            geom.contype, geom.conaffinity = 0, 0
    for c in colliders:
        spec.body(c["body"]).add_geom(
            name=c["name"], type=mujoco.mjtGeom(c["type"]), size=c["size"], pos=c["pos"], quat=c["quat"],
            contype=1, conaffinity=1 if self_collision else 0, condim=3, friction=[1.0, 0.005, 0.0001],
            group=COLLIDER_GROUP, rgba=[0.3, 0.6, 1.0, 0.35], margin=0.0,
        )  # fmt: skip
    if self_collision:
        # MyoFullBody's own explicit pairs act on its (now visual) geometry; the plant's
        # contact model replaces them.
        for pair in list(spec.pairs):
            spec.delete(pair)
        for a, b in ADJACENT_HOSTS:
            spec.add_exclude(bodyname1=a, bodyname2=b)


# --------------------------------------------------------------------------- #
# Fitting the table
# --------------------------------------------------------------------------- #
def sample_poses(model: mujoco.MjModel, n: int, seed: int = 0) -> np.ndarray:
    """``(n, nq)`` random poses: independent hinges uniform in range, couplings exact."""
    from myosuite.integrations.yoga.kinematics import ReducedKinematics

    rk = ReducedKinematics(model)
    rng = np.random.default_rng(seed)
    out = np.empty((n, model.nq))
    for i in range(n):
        q = model.qpos0.copy()
        for adr, lo, hi in zip(
            rk.indep_hinge_qadr, rk.indep_hinge_lo, rk.indep_hinge_hi
        ):
            q[adr] = rng.uniform(lo, hi)
        out[i] = rk.apply_couplings(q)
    return out


def actuator_lengths(model: mujoco.MjModel, poses: np.ndarray) -> np.ndarray:
    """``(n, nu)`` actuator (musculotendon) lengths at each pose."""
    data = mujoco.MjData(model)
    out = np.empty((len(poses), model.nu))
    for i, q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_tendon(model, data)
        mujoco.mj_transmission(model, data)
        out[i] = data.actuator_length
    return out


def myo_surface_lengths(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Surface lengths of MyoFullBody at rest: foot, elbow-to-middle-fingertip, crown above the shoulders."""
    jc = myo_joint_centres(model, data)
    out = {}
    for s in SIDES:
        foot = [f"calcn_{s}", f"toes_{s}"]
        out[f"foot_len_{s}"] = collider_extent(
            model, data, foot, MYO_FORWARD
        ) + collider_extent(model, data, foot, -MYO_FORWARD)
        axis = jc[f"hand_{s}"] - jc[f"elbow_{s}"]
        tip = collider_extent(model, data, [f"3distph_{s}"], axis) - float(
            jc[f"elbow_{s}"] @ (axis / np.linalg.norm(axis))
        )
        out[f"reach_{s}"] = tip
    gh = 0.5 * (jc["shoulder_l"] + jc["shoulder_r"])
    out["crown_above_gh"] = collider_extent(model, data, ["head"], MYO_UP) - float(
        gh[2]
    )
    return out


def her_surface_lengths(t: dict[str, np.ndarray]) -> dict[str, float]:
    """The same surface lengths on her template (plant frame, x forward)."""
    J = t["joints"]
    elbow = {"l": J[18], "r": J[19]}
    out = {}
    for s in SIDES:
        out[f"foot_len_{s}"] = float(t[f"bigtoe_{s}"][0] - t[f"heel_{s}"][0])
        out[f"reach_{s}"] = float(np.linalg.norm(t[f"middletip_{s}"] - elbow[s]))
    out["crown_above_gh"] = float(t["crown"][2] - 0.5 * (J[16][2] + J[17][2]))
    return out


def muscle_length_ratios(
    table: SubjectTable, finger_flexors: bool = False, n_poses: int = 400
) -> dict[str, float]:
    """Musculotendon-length ratio (scaled / stock geometry) of every muscle.

    Measured over random poses of the stock model; ``lengthrange`` times the ratio keeps
    each muscle's calibration (optimal fibre and tendon slack length) in normalised units.

    Args:
        table: Subject table (only its geometry is used).
        finger_flexors: Include the extrinsic finger flexors (:data:`FINGER_FLEXORS`).
        n_poses: Number of random poses.
    """
    model = compile_spec(base_fullbody_spec(finger_flexors))
    geo_spec = base_fullbody_spec(finger_flexors)
    _scale_geometry(geo_spec, compile_spec(geo_spec.copy()), table)
    scaled = compile_spec(geo_spec)
    poses = sample_poses(model, n_poses)
    L0, L1 = actuator_lengths(model, poses), actuator_lengths(scaled, poses)
    ratio = (L0 * L1).sum(0) / (L0 * L0).sum(0)
    return {
        model.actuator(i).name: float(ratio[i])
        for i in range(model.nu)
        if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
    }


def upgrade_table(
    table: SubjectTable,
    wrist_extension_deg: float = R3_WRIST_EXTENSION_DEG,
    undo_shoulder_edit: bool = False,
    n_poses: int = 400,
) -> SubjectTable:
    """The round-3 plant: the hand and wrist fixes of the strength audit.

    See ``notes/muscle_yoga/strength_audit.md`` §9: extrinsic finger flexors kept (their
    ``lengthrange`` scaled like every other muscle's), wrist paths repaired beyond ±45°,
    wrist extension opened to *wrist_extension_deg*, stiff wrist and elbow end-ranges, and
    the supraspinatus' active force restored. Passive compliance is not refitted here: run
    ``statics --write-compliant`` on clips retargeted to the new body.

    Args:
        table: The table to upgrade.
        wrist_extension_deg: Wrist extension limit (deg).
        undo_shoulder_edit: Also restore TMIN, TMAJ, INFSP and LAT2 to their MoBL-ARMS
            forces (:data:`SHOULDER_EDIT_UNDO`).
        n_poses: Random poses for the finger flexors' length ratios.
    """
    ratio = muscle_length_ratios(table, finger_flexors=True, n_poses=n_poses)
    stock = compile_spec(base_fullbody_spec(finger_flexors=True))
    lengthrange = dict(table.lengthrange)
    for name in (_arm_name(k, s) for k in FINGER_FLEXORS for s in SIDES):
        lengthrange[name] = (
            stock.actuator_lengthrange[stock.actuator(name).id] * ratio[name]
        ).tolist()
    ranges = dict(table.joint_ranges)
    for s in SIDES:
        ranges[f"flexion_{s}"] = [
            -float(np.radians(wrist_extension_deg)),
            float(ranges[f"flexion_{s}"][1]),
        ]
    stiff = [f"{j}_{s}" for j in ("flexion", "elbow_flex") for s in SIDES]
    factors = {**SUPSP_FIX, **(SHOULDER_EDIT_UNDO if undo_shoulder_edit else {})}
    return replace(
        table, finger_flexors=True, lengthrange=lengthrange, joint_ranges=ranges,
        wrist_obstacles={k: list(v) for k, v in WRIST_OBSTACLES.items()},
        limit_solref={j: list(STIFF_LIMIT_SOLREF) for j in stiff},
        limit_solimp={j: list(STIFF_LIMIT_SOLIMP) for j in stiff},
        strength_scale={_arm_name(k, s): v for k, v in factors.items() for s in SIDES},
    )  # fmt: skip


def fit_subject_table(
    plant_xml: Path,
    mosh_pkl: Path,
    smplx_npz: Path,
    neck: bool = True,
    n_poses: int = 400,
) -> SubjectTable:
    """Measure both skeletons and calibrate the subject table.

    Scale sources: joint-centre distances where SMPL-X and MyoFullBody define the joint
    alike (hip, knee, ankle, shoulder, elbow; hip and shoulder width); surface distances
    where they do not (SMPL-X places the wrist ~12 cm and the "toe" joint behind the
    anatomical joints): foot length heel-to-big-toe, elbow-to-middle-fingertip reach for
    the forearm and hand, and crown height above the shoulders for the head.

    Args:
        plant_xml: The performer's plant MJCF (``smpl_yogi03596_v2.xml``): segment masses.
        mosh_pkl: One of her MoSh++ stage-II fits (template).
        smplx_npz: ``SMPLX_FEMALE.npz``.
        neck: Add the torque-driven neck.
        n_poses: Random poses used to measure musculotendon-length ratios.

    Returns:
        The fitted table, with the audit numbers in ``measured``.
    """
    base = base_fullbody_spec()
    model = compile_spec(base.copy())
    data = _rest(model)
    jc = myo_joint_centres(model, data)
    myo_len = segment_lengths(jc)
    her_t = her_template(mosh_pkl, smplx_npz)
    her_len = segment_lengths(
        plant_joint_centres(_plant_rest_from_joints(her_t["joints"]))
    )
    myo_surf, her_surf = myo_surface_lengths(model, data), her_surface_lengths(her_t)
    scale: dict[str, float] = {}
    for s in SIDES:
        for seg in ("thigh", "shank", "upperarm"):
            scale[f"{seg}_{s}"] = her_len[f"{seg}_{s}"] / myo_len[f"{seg}_{s}"]
        reach = her_surf[f"reach_{s}"] / myo_surf[f"reach_{s}"]
        scale[f"forearm_{s}"] = scale[f"hand_{s}"] = reach
    foot = 0.5 * sum(
        her_surf[f"foot_len_{s}"] / myo_surf[f"foot_len_{s}"] for s in SIDES
    )
    for s in SIDES:
        scale[f"foot_{s}"] = foot
    hip_w = her_len["hip_width"] / myo_len["hip_width"]
    # Shoulder width through the clavicle: scaling it moves the girdle by (k-1) * d_clav.
    clav = {}
    half_target = 0.5 * her_len["shoulder_width"]
    mid = 0.5 * (jc["shoulder_l"] + jc["shoulder_r"])
    for s in SIDES:
        d = (
            data.xpos[model.body(f"clavphant_{s}").id]
            - data.xpos[model.body(f"clavicle_{s}").id]
        )
        lat_now = abs(float((jc[f"shoulder_{s}"] - mid) @ MYO_LATERAL))
        clav[s] = 1.0 + (half_target - lat_now) / abs(float(d @ MYO_LATERAL))
    head_lift = her_surf["crown_above_gh"] - myo_surf["crown_above_gh"]
    crown_myo = myo_surf["crown_above_gh"]
    pmodel = mujoco.MjModel.from_xml_path(str(plant_xml))
    plant_mass = {n: float(pmodel.body_mass[pmodel.body(n).id]) for n in PLANT_BODIES}
    group_mass = {
        g: sum(plant_mass[n] for n in names) for g, names in PLANT_MASS_GROUPS.items()
    }
    table = SubjectTable(
        segment_scale=scale, hip_width_scale=hip_w, clavicle_scale=clav, head_lift=head_lift,
        group_mass=group_mass, neck=neck,
        joint_ranges={k: list(np.radians(v)) for k, v in YOGA_RANGES_DEG.items()},
        coupling_extension={f"knee_angle_{s}": float(np.radians(KNEE_COUPLING_FIT_DEG)) for s in SIDES},
    )  # fmt: skip
    ratio = muscle_length_ratios(table, n_poses=n_poses)
    table.lengthrange = {
        n: (model.actuator_lengthrange[model.actuator(n).id] * r).tolist()
        for n, r in ratio.items()
    }
    base_group: dict[str, float] = {}
    for b in range(1, model.nbody):
        g = myo_body_groups(model)[model.body(b).name]
        base_group[g] = base_group.get(g, 0.0) + float(model.body_mass[b])
    table.measured = {
        "her_segments_m": her_len, "myo_segments_m": myo_len,
        "her_surface_m": her_surf, "myo_surface_m": myo_surf,
        "her_group_mass_kg": group_mass, "myo_group_mass_kg": base_group,
        "myo_crown_above_gh_m": crown_myo, "her_crown_above_gh_m": crown_myo + head_lift,
        "her_stature_m": float(her_t["crown"][2] - her_t["sole"][2]),
        "muscle_length_ratio": {"min": float(min(ratio.values())), "median": float(np.median(list(ratio.values()))), "max": float(max(ratio.values()))},
    }  # fmt: skip
    return table


def _plant_rest_from_joints(J: np.ndarray) -> np.ndarray:
    """Plant rest body origins (PLANT_BODIES order) from her 55 SMPL-X joints."""
    idx = {
        "Pelvis": 0, "L_Hip": 1, "L_Knee": 4, "L_Ankle": 7, "L_Toe": 10, "R_Hip": 2, "R_Knee": 5,
        "R_Ankle": 8, "R_Toe": 11, "Torso": 3, "Spine": 6, "Chest": 9, "Neck": 12, "Head": 15,
        "L_Thorax": 13, "L_Shoulder": 16, "L_Elbow": 18, "L_Wrist": 20, "R_Thorax": 14,
        "R_Shoulder": 17, "R_Elbow": 19, "R_Wrist": 21,
    }  # fmt: skip
    out = np.zeros((len(PLANT_BODIES), 3))
    for i, n in enumerate(PLANT_BODIES):
        if n in idx:
            out[i] = J[idx[n]]
    out[PLANT_BODIES.index("L_Hand")] = J[[25, 28, 31, 34]].mean(0)
    out[PLANT_BODIES.index("R_Hand")] = J[[40, 43, 46, 49]].mean(0)
    return out


def main() -> None:
    """Fit the subject table (or upgrade one), write it and print the audit."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plant-xml", type=Path)
    ap.add_argument("--mosh-pkl", type=Path, help="any of her MoSh++ stage-II fits")
    ap.add_argument("--smplx", type=Path, help="SMPLX_FEMALE.npz")
    ap.add_argument("--out", type=Path, required=True, help="subject table JSON")
    ap.add_argument(
        "--upgrade", type=Path, default=None,
        help="instead of fitting: write this table upgraded to the round-3 plant (upgrade_table)",
    )  # fmt: skip
    ap.add_argument("--wrist-extension-deg", type=float, default=R3_WRIST_EXTENSION_DEG)
    ap.add_argument("--undo-shoulder-edit", action="store_true")
    ap.add_argument("--no-neck", action="store_true")
    ap.add_argument(
        "--keep-myo-colliders",
        action="store_true",
        help="do not transplant the plant's colliders",
    )
    ap.add_argument(
        "--no-self-collision",
        action="store_true",
        help="keep MyoFullBody's explicit pairs instead of plant-style self-collision",
    )
    ap.add_argument(
        "--xml-out", type=Path, default=None, help="also write the compiled model XML"
    )
    args = ap.parse_args()
    if args.upgrade is not None:
        table = upgrade_table(
            SubjectTable.load(args.upgrade),
            args.wrist_extension_deg,
            args.undo_shoulder_edit,
        )
        table.save(args.out)
        model = compile_spec(build_subject_spec(table))
        print(f"wrote {args.out}: {sum(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)} muscles, "
              f"strength_scale {table.strength_scale}, stiff limits {sorted(table.limit_solref)}")  # fmt: skip
        return
    if None in (args.plant_xml, args.mosh_pkl, args.smplx):
        ap.error("--plant-xml, --mosh-pkl and --smplx are required to fit a table")
    table = fit_subject_table(
        args.plant_xml, args.mosh_pkl, args.smplx, neck=not args.no_neck
    )
    if not args.keep_myo_colliders:
        from myosuite.integrations.yoga.retarget import transplant_plant_colliders

        model = compile_spec(build_subject_spec(table))
        table.colliders = transplant_plant_colliders(
            model, args.plant_xml, her_template(args.mosh_pkl, args.smplx)
        )
        table.self_collision = not args.no_self_collision
    table.save(args.out)
    m = table.measured
    print(f"{'segment':16s} {'her (m)':>8s} {'myo (m)':>8s} {'ratio':>6s}")
    for key in ("segments_m", "surface_m"):
        for k in m[f"her_{key}"]:
            h, y = m[f"her_{key}"][k], m[f"myo_{key}"][k]
            print(f"{k:16s} {h:8.4f} {y:8.4f} {h / y:6.3f}")
    print("scales", {k: round(v, 4) for k, v in table.segment_scale.items()})
    print("her stature", round(m["her_stature_m"], 4))
    print(
        "clavicle scale",
        table.clavicle_scale,
        "hip width scale",
        round(table.hip_width_scale, 4),
    )
    print("head lift (m)", round(table.head_lift, 4))
    for g in m["her_group_mass_kg"]:
        print(
            f"mass {g:12s} her {m['her_group_mass_kg'][g]:6.2f}  myo {m['myo_group_mass_kg'][g]:6.2f}"
        )
    print("muscle length ratio", m["muscle_length_ratio"])
    if args.xml_out is not None:
        spec = build_subject_spec(table)
        spec.compile()
        args.xml_out.write_text(spec.to_xml())


if __name__ == "__main__":
    main()
