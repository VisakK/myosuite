# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Contact-aware retarget of yoga references onto the subject MyoFullBody.

The references move her 24-body plant skeleton; MyoFullBody has a different DOF
structure (coupled spine and shoulder girdle, 1-DOF knee, no thoracic joint), so
angles cannot be copied. The retarget is OpenSim-style marker inverse kinematics:

1. **Registration.** A static solve puts MyoFullBody in her rest (T) pose. Every
   landmark (MyoFullBody joint centres and orientation points) is then registered
   rigidly on the plant body that carries it, at its T-pose offset.
2. **Per frame.** Projected Levenberg-Marquardt over the independent coordinates
   (couplings applied exactly, joint ranges as a box) minimises weighted landmark
   errors plus floor terms: no collider below the floor, and every plant body that
   touches the floor in the reference pulls its MyoFullBody counterpart onto it.

Output is a MuscleMimic clip ``.npz`` at the control rate (``qpos``, ``qvel``,
``site_xpos``, ``site_names``, ``frequency``) plus a metrics ``.json``.

CLI::

    python -m myosuite.integrations.yoga.retarget --table subject_table.json \
        --mosh-pkl <fit.pkl> --smplx SMPLX_FEMALE.npz --plant-xml <plant.xml> \
        --refs <dir of reference npz> --out <dir> [--clips substr ...] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
)
from myosuite.integrations.yoga import subject_body as sb
from myosuite.integrations.yoga.kinematics import ReducedKinematics
from myosuite.integrations.yoga.reference_io import (
    PLANT_BODIES,
    YogaReference,
    load_reference,
    plant_body_lowest,
    plant_colliders,
    quat_to_mat,
)

PI = {n: i for i, n in enumerate(PLANT_BODIES)}
CONTROL_HZ = 100.0
FLOOR_CONTACT_M = 0.015  # plant collider within this of the floor = supporting contact
# Weak pull to neutral on DOFs the landmarks leave nearly redundant (axial twists, foot roll).
PRIOR_JOINTS = (
    "shoulder_rot",
    "pro_sup",
    "deviation",
    "subtalar_angle",
    "mtp_angle",
    "hip_rotation",
)
W_PRIOR = 0.02
W_BRACE = (
    3.0  # weight of body-body overlap beyond the allowance (with --brace-allowance)
)

# Plant body -> MyoFullBody body that carries its transplanted collider.
PLANT_COLLIDER_HOST: dict[str, str] = {
    "Pelvis": "pelvis", "Torso": "lumbar3", "Spine": "lumbar1", "Chest": "torso", "Neck": "head", "Head": "head",
    **{f"{S}_{seg}": f"{host}_{s}" for s, S in sb.PLANT_SIDE.items() for seg, host in (
        ("Hip", "femur"), ("Knee", "tibia"), ("Ankle", "calcn"), ("Toe", "toes"), ("Thorax", "clavicle"),
        ("Shoulder", "humerus"), ("Elbow", "ulna"), ("Wrist", "lunate"), ("Hand", "lunate"))},
}  # fmt: skip

# Plant body -> MyoFullBody bodies whose colliders realise that contact.
CONTACT_BODIES: dict[str, tuple[str, ...]] = {
    "Pelvis": ("pelvis",), "Torso": ("lumbar3", "torso"), "Spine": ("lumbar1", "torso"),
    "Chest": ("torso", "thorax"), "Neck": ("head", "neck"), "Head": ("head",),
    **{f"{S}_Thorax": (f"clavicle_{s}",) for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Hip": (f"femur_{s}",) for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Knee": (f"tibia_{s}",) for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Ankle": (f"calcn_{s}",) for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Toe": (f"calcn_{s}", f"toes_{s}") for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Shoulder": (f"humerus_{s}",) for s, S in sb.PLANT_SIDE.items()},
    **{f"{S}_Elbow": (f"ulna_{s}", f"radius_{s}") for s, S in sb.PLANT_SIDE.items()},
}  # fmt: skip


def transplant_plant_colliders(
    model: mujoco.MjModel, plant_xml: Path, her: dict[str, np.ndarray]
) -> list[dict]:
    """The plant's colliders expressed in MyoFullBody body frames (subject-table entries).

    Each plant primitive is moved onto its host bone (:data:`PLANT_COLLIDER_HOST`) with
    the registration T-pose: there the plant body frames are the world axes, so the
    primitive's world pose in the plant rest pose, expressed in the host's T-pose frame,
    is its pose on the host. The retarget aligns the registered landmarks, so in every
    frame the transplanted primitives sit where the plant's did.

    Args:
        model: Subject model with MyoFullBody's own colliders (registration geometry).
        plant_xml: Plant MJCF.
        her: Her template landmarks.
    """
    rt = Retargeter(model, plant_xml, her)
    d = mujoco.MjData(model)
    d.qpos[:] = rt.tpose_qpos
    mujoco.mj_forward(model, d)
    cols = rt.colliders
    out = []
    for g, b in enumerate(cols.body_index):
        pb = PLANT_BODIES[b]
        host = model.body(PLANT_COLLIDER_HOST[pb]).id
        Rm = d.xmat[host].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (Rm.T @ cols.mat[g]).reshape(-1))
        out.append({
            "name": f"plant_{pb}_{g}", "body": model.body(host).name, "type": int(cols.geom_type[g]),
            "size": cols.size[g].tolist(), "pos": (Rm.T @ (cols.rest_body_pos[b] + cols.pos[g] - d.xpos[host])).tolist(),
            "quat": quat.tolist(),
        })  # fmt: skip
    return out


def registration_model(table: sb.SubjectTable) -> mujoco.MjModel:
    """The subject model with MyoFullBody's own colliders (landmarks are defined on them)."""
    import dataclasses

    return sb.compile_spec(
        sb.build_subject_spec(dataclasses.replace(table, colliders=[]))
    )


@dataclass(frozen=True)
class Landmark:
    """A MyoFullBody point and the plant body that carries its target."""

    name: str
    myo_body: str
    plant_body: str
    weight: float


def landmark_set() -> list[Landmark]:
    """Joint centres (weight 1) and orientation points of every segment."""
    L: list[Landmark] = []
    for s, S in sb.PLANT_SIDE.items():
        L += [
            Landmark(f"hip_{s}", f"femur_{s}", f"{S}_Hip", 1.0),
            Landmark(f"knee_{s}", f"tibia_{s}", f"{S}_Knee", 1.0),
            Landmark(f"ankle_{s}", f"talus_{s}", f"{S}_Ankle", 1.0),
            Landmark(f"shoulder_{s}", f"humerus_{s}", f"{S}_Shoulder", 1.0),
            Landmark(f"elbow_{s}", f"ulna_{s}", f"{S}_Elbow", 1.0),
            Landmark(f"wrist_{s}", f"lunate_{s}", f"{S}_Elbow", 0.7),
            Landmark(f"thigh_ant_{s}", f"femur_{s}", f"{S}_Hip", 0.2),
            Landmark(f"shank_ant_{s}", f"tibia_{s}", f"{S}_Knee", 0.5),
            Landmark(f"heel_{s}", f"calcn_{s}", f"{S}_Ankle", 0.7),
            Landmark(f"foot_lat_{s}", f"calcn_{s}", f"{S}_Ankle", 0.5),
            Landmark(f"mtp_{s}", f"toes_{s}", f"{S}_Ankle", 0.7),
            Landmark(f"toe_tip_{s}", f"toes_{s}", f"{S}_Toe", 0.3),
            Landmark(f"uarm_post_{s}", f"humerus_{s}", f"{S}_Shoulder", 0.1),
            Landmark(f"farm_dors_{s}", f"radius_{s}", f"{S}_Elbow", 0.3),
            Landmark(f"mcp2_{s}", f"2proxph_{s}", f"{S}_Wrist", 0.6),
            Landmark(f"mcp5_{s}", f"5proxph_{s}", f"{S}_Wrist", 0.6),
            Landmark(f"fingertip_{s}", f"3distph_{s}", f"{S}_Hand", 0.2),
            Landmark(f"chest_lat_{s}", "thorax", "Chest", 0.5),
        ]
    L += [
        Landmark("pelvis_ant", "pelvis", "Pelvis", 0.7),
        Landmark("pelvis_sup", "pelvis", "Pelvis", 0.7),
        Landmark("chest_ant", "thorax", "Chest", 0.6),
        Landmark("chest_sup", "thorax", "Chest", 0.6),
        Landmark("head_sup", "head", "Head", 0.6),
        Landmark("head_ant", "head", "Head", 0.6),
        Landmark("head_lat", "head", "Head", 0.4),
    ]
    return L


def _myo_points(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """World position of every landmark on a forwarded MyoFullBody, in the T-pose.

    Joint centres are body origins; orientation points are offsets along the T-pose
    world axes of the plant frame (x forward, y left, z up), so they are only
    meaningful when ``data`` holds the registration T-pose.
    """
    x = lambda b: data.xpos[model.body(b).id].copy()  # noqa: E731
    F, LFT, U = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    p: dict[str, np.ndarray] = {}
    for s in sb.SIDES:
        side = 1.0 if s == "l" else -1.0
        p[f"hip_{s}"], p[f"knee_{s}"], p[f"ankle_{s}"] = (
            x(f"femur_{s}"),
            x(f"tibia_{s}"),
            x(f"talus_{s}"),
        )
        p[f"shoulder_{s}"], p[f"elbow_{s}"], p[f"wrist_{s}"] = (
            x(f"humerus_{s}"),
            x(f"ulna_{s}"),
            x(f"lunate_{s}"),
        )
        p[f"thigh_ant_{s}"] = p[f"knee_{s}"] + 0.10 * F + 0.08 * U
        p[f"shank_ant_{s}"] = p[f"ankle_{s}"] + 0.08 * F + 0.08 * U
        foot = [f"calcn_{s}", f"toes_{s}"]
        p[f"heel_{s}"] = np.array(
            [
                -sb.collider_extent(model, data, foot, -F),
                p[f"ankle_{s}"][1],
                p[f"ankle_{s}"][2] - 0.04,
            ]
        )
        lat = side * sb.collider_extent(model, data, [f"calcn_{s}"], side * LFT)
        p[f"foot_lat_{s}"] = np.array(
            [p[f"ankle_{s}"][0] + 0.06, lat, p[f"ankle_{s}"][2] - 0.04]
        )
        p[f"mtp_{s}"] = x(f"toes_{s}")
        p[f"toe_tip_{s}"] = np.array(
            [
                sb.collider_extent(model, data, [f"toes_{s}"], F),
                p[f"mtp_{s}"][1],
                p[f"mtp_{s}"][2],
            ]
        )
        p[f"uarm_post_{s}"] = p[f"elbow_{s}"] - 0.08 * F + 0.04 * side * -LFT
        p[f"farm_dors_{s}"] = p[f"wrist_{s}"] + 0.06 * U + 0.04 * side * -LFT
        p[f"mcp2_{s}"], p[f"mcp5_{s}"] = x(f"2proxph_{s}"), x(f"5proxph_{s}")
        axis = p[f"mcp2_{s}"] + p[f"mcp5_{s}"] - 2 * p[f"wrist_{s}"]
        axis /= np.linalg.norm(axis)
        reach = sb.collider_extent(model, data, [f"3distph_{s}"], axis)
        mid = 0.5 * (p[f"mcp2_{s}"] + p[f"mcp5_{s}"])
        p[f"fingertip_{s}"] = mid + (reach - mid @ axis) * axis
    th = x("thorax")
    gh = 0.5 * (p["shoulder_l"] + p["shoulder_r"])
    for s in sb.SIDES:
        p[f"chest_lat_{s}"] = np.array(
            [gh[0], (1.0 if s == "l" else -1.0) * 0.11 + gh[1], gh[2] - 0.12]
        )
    pel = x("pelvis")
    p["pelvis_ant"] = pel + 0.12 * F
    p["pelvis_sup"] = pel + 0.12 * U
    p["chest_ant"] = np.array([gh[0] + 0.12, gh[1], gh[2] - 0.10])
    p["chest_sup"] = np.array([th[0] - 0.02, gh[1], gh[2] + 0.04])
    crown = sb.collider_extent(model, data, ["head"], U)
    p["head_sup"] = np.array([gh[0], gh[1], crown])
    p["head_ant"] = np.array([gh[0] + 0.12, gh[1], crown - 0.12])
    p["head_lat"] = np.array([gh[0], gh[1] + 0.09, crown - 0.10])
    return p


class Retargeter:
    """Registered marker IK of plant references onto a MyoFullBody model.

    Args:
        model: The subject MyoFullBody (compiled).
        plant_xml: Plant MJCF (colliders, for the reference contact states).
        her: Her template landmarks (:func:`subject_body.her_template`).
        registration: Model with MyoFullBody's own colliders on which the landmarks
            are defined (same kinematics); ``model`` itself when ``None``.
        brace_allowance: With self-collision, the overlap (m) two primitives may keep
            before an IK term pushes them apart; ``None`` ignores body-body contact.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        plant_xml: Path,
        her: dict[str, np.ndarray],
        registration: mujoco.MjModel | None = None,
        brace_allowance: float | None = None,
    ) -> None:
        self.model = model
        self.brace_allowance = brace_allowance
        self._reg = model if registration is None else registration
        self.data = mujoco.MjData(model)
        self.rk = ReducedKinematics(model)
        self.colliders = plant_colliders(plant_xml)
        self.landmarks = landmark_set()
        self.her = her
        self._body_id = {
            b: model.body(b).id for b in {lm.myo_body for lm in self.landmarks}
        }
        self._contact_geoms = {
            k: np.array([g for g in range(model.ngeom) if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name in v])
            for k, v in CONTACT_BODIES.items()
        }  # fmt: skip
        hand = {
            s: [
                model.body(n).name
                for n in range(model.nbody)
                if self._in_subtree(n, f"lunate_{s}")
            ]
            for s in sb.SIDES
        }
        for s, S in sb.PLANT_SIDE.items():
            g = np.array(
                [
                    i
                    for i in range(model.ngeom)
                    if model.geom_contype[i]
                    and model.body(model.geom_bodyid[i]).name in hand[s]
                ]
            )
            self._contact_geoms[f"{S}_Wrist"] = g
            self._contact_geoms[f"{S}_Hand"] = g
        prior = [
            j
            for j in range(model.njnt)
            if model.joint(j).name.rsplit("_", 1)[0] in PRIOR_JOINTS
        ]
        self._prior_qadr = np.array([model.jnt_qposadr[j] for j in prior])
        self._prior_dof = np.array([model.jnt_dofadr[j] for j in prior])
        self.all_colliders = np.array(
            [
                g
                for g in range(model.ngeom)
                if model.geom_contype[g] and model.geom_bodyid[g] != 0
            ]
        )
        self.tpose_qpos, self.offsets = self._register()

    def _in_subtree(self, b: int, root: str) -> bool:
        r = self.model.body(root).id
        while b not in (0, r):
            b = int(self.model.body_parentid[b])
        return b == r

    # ----------------------------------------------------------------- registration
    def _plant_rest(self) -> dict[str, np.ndarray]:
        return {n: self.colliders.rest_body_pos[i] for i, n in enumerate(PLANT_BODIES)}

    def _tpose_targets(self) -> list[tuple[str, str, np.ndarray, float]]:
        """(name, myo body, rest-frame target, weight) for the static T-pose solve."""
        rest, her = self._plant_rest(), self.her
        J = her["joints"]
        t: list[tuple[str, str, np.ndarray, float]] = []
        for s, S in sb.PLANT_SIDE.items():
            t += [
                (f"knee_{s}", f"tibia_{s}", rest[f"{S}_Knee"], 1.0),
                (f"ankle_{s}", f"talus_{s}", rest[f"{S}_Ankle"], 1.0),
                (f"elbow_{s}", f"ulna_{s}", rest[f"{S}_Elbow"], 1.0),
                (f"shoulder_{s}", f"humerus_{s}", rest[f"{S}_Shoulder"], 1.0),
                (f"heel_{s}", f"calcn_{s}", her[f"heel_{s}"], 0.5),
                (f"bigtoe_{s}", f"toes_{s}", her[f"bigtoe_{s}"], 0.5),
                (f"mcp2_{s}", f"2proxph_{s}", J[25 if s == "l" else 40], 0.5),
                (f"mcp5_{s}", f"5proxph_{s}", J[31 if s == "l" else 46], 0.5),
                (f"fingertip_{s}", f"3distph_{s}", her[f"middletip_{s}"], 0.5),
            ]
        return t

    def _register(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Solve the T-pose and register every landmark on its plant body."""
        m = self._reg
        d = mujoco.MjData(m)
        rest = self._plant_rest()
        q = m.qpos0.copy()
        q[3:7] = [
            np.cos(np.pi / 4),
            0.0,
            0.0,
            np.sin(np.pi / 4),
        ]  # face +x (plant forward)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        hip_mid = 0.5 * (d.xpos[m.body("femur_l").id] + d.xpos[m.body("femur_r").id])
        q[:3] += 0.5 * (rest["L_Hip"] + rest["R_Hip"]) - hip_mid
        free = [
            j
            for j in range(m.njnt)
            if m.joint(j).name.split("_")[0]
            in (
                "elv",
                "shoulder",
                "elbow",
                "pro",
                "deviation",
                "flexion",
                "hip",
                "knee",
                "ankle",
            )
            and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        free_dof = {int(m.jnt_dofadr[j]) for j in free}
        mask = np.array([d_ in free_dof for d_ in self.rk.indep_dof])
        targets = self._tpose_targets()
        bodies = [m.body(b).id for _, b, _, _ in targets]
        # Hand/foot surface targets use Myo's own geometric landmarks (collider extremes).
        q = self._solve_static(q, targets, bodies, mask)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        pts = _myo_points(m, d)
        offsets = {lm.name: pts[lm.name] - rest[lm.plant_body] for lm in self.landmarks}
        self._local = {}
        for lm in self.landmarks:
            b = self._body_id[lm.myo_body]
            R = d.xmat[b].reshape(3, 3)
            self._local[lm.name] = R.T @ (pts[lm.name] - d.xpos[b])
        return q, offsets

    def _static_points(
        self, targets: list[tuple], bodies: list[int], d: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Current world points of the static-solve targets and their Jacobians."""
        m = self._reg
        P, Jp = [], []
        F = np.array([1.0, 0, 0])
        for (name, body, _, _), b in zip(targets, bodies):
            base = name.split("_")[0]
            if base in ("heel", "bigtoe", "fingertip"):
                if base == "heel":
                    u = -F
                elif base == "bigtoe":
                    u = F
                else:
                    u = d.xpos[b] - d.xpos[m.body(f"lunate_{name[-1]}").id]
                u = u / np.linalg.norm(u)
                p = self._support_point(b, u, d)
            else:
                p = d.xpos[b].copy()
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, p, b)
            P.append(p)
            Jp.append(jp)
        return np.array(P), np.array(Jp)

    def _support_point(self, body: int, u: np.ndarray, d: mujoco.MjData) -> np.ndarray:
        """World point of *body*'s colliders furthest along *u* (registration geometry)."""
        m = self._reg
        best, arg = -np.inf, None
        for g in np.nonzero((m.geom_bodyid == body) & (m.geom_contype > 0))[0]:
            p = _support(
                m.geom_type[g],
                d.geom_xpos[g],
                d.geom_xmat[g].reshape(3, 3),
                m.geom_size[g],
                u,
            )
            if p @ u > best:
                best, arg = p @ u, p
        return arg

    def _solve_static(
        self,
        q: np.ndarray,
        targets: list[tuple],
        bodies: list[int],
        mask: np.ndarray,
        iters: int = 200,
    ) -> np.ndarray:
        m = self._reg
        d = mujoco.MjData(m)
        W = np.repeat([w for *_, w in targets], 3)
        tgt = np.concatenate([t for _, _, t, _ in targets])
        lam = 1e-2
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            P, Jp = self._static_points(targets, bodies, d)
            r = W * (P.reshape(-1) - tgt)
            J = (W[:, None] * Jp.reshape(-1, m.nv)) @ self.rk.velocity_map(q)[:, mask]
            dq = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), -J.T @ r)
            v = np.zeros(m.nv)
            v += self.rk.velocity_map(q)[:, mask] @ dq
            q = q.copy()
            mujoco.mj_integratePos(m, q, v, 1.0)
            q = self.rk.clamp(q)
            if np.abs(dq).max() < 1e-7:
                break
        return q

    # ----------------------------------------------------------------- frame solve
    def targets(self, ref: YogaReference) -> np.ndarray:
        """``(T, L, 3)`` landmark targets from the plant body poses."""
        R = quat_to_mat(ref.body_quat)
        out = np.empty((ref.n_frames, len(self.landmarks), 3))
        for i, lm in enumerate(self.landmarks):
            b = PI[lm.plant_body]
            out[:, i] = ref.body_pos[:, b] + np.einsum(
                "tij,j->ti", R[:, b], self.offsets[lm.name]
            )
        return out

    def _residual(
        self, q: np.ndarray, tgt: np.ndarray, contact: list[str], q_prev: np.ndarray | None,
        w_contact: float, w_pen: float, w_reg: float,
    ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
        m, d = self.model, self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, jac = [], []
        jb = {}
        for name, b in self._body_id.items():
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, b)
            jb[b] = (jp, jr)
        for i, lm in enumerate(self.landmarks):
            b = self._body_id[lm.myo_body]
            R = d.xmat[b].reshape(3, 3)
            off = R @ self._local[lm.name]
            p = d.xpos[b] + off
            jp, jr = jb[b]
            J = jp - _skew(off) @ jr
            rows.append(lm.weight * (p - tgt[i]))
            jac.append(lm.weight * J)
        geoms = self.all_colliders
        low = _lowest_points(
            m.geom_type[geoms],
            d.geom_xpos[geoms],
            d.geom_xmat[geoms].reshape(-1, 3, 3),
            m.geom_size[geoms],
        )
        pen = np.nonzero(low[:, 2] < 0.0)[0]
        for k in pen:
            g = geoms[k]
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, low[k], int(m.geom_bodyid[g]))
            rows.append(np.array([w_pen * low[k, 2]]))
            jac.append(w_pen * jp[2:3])
        gindex = {g: k for k, g in enumerate(geoms)}
        for pb in contact:
            gs = self._contact_geoms.get(pb)
            if gs is None or len(gs) == 0:
                continue
            ks = [gindex[g] for g in gs]
            k = ks[int(np.argmin(low[ks, 2]))]
            if low[k, 2] < 0.0:
                continue  # already penalised as penetration
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, low[k], int(m.geom_bodyid[geoms[k]]))
            rows.append(np.array([w_contact * low[k, 2]]))
            jac.append(w_contact * jp[2:3])
        if self.brace_allowance is not None:
            # Self-collision: primitives may overlap by the allowance (soft tissue), not more.
            mujoco.mj_collision(m, d)
            for c in d.contact[: d.ncon]:
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if b1 == 0 or b2 == 0 or c.dist >= -self.brace_allowance:
                    continue
                j1, j2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, j1, None, c.pos, b1)
                mujoco.mj_jac(m, d, j2, None, c.pos, b2)
                rows.append(np.array([W_BRACE * (c.dist + self.brace_allowance)]))
                jac.append(W_BRACE * (np.asarray(c.frame[:3]) @ (j2 - j1))[None])
        rows.append(W_PRIOR * q[self._prior_qadr])
        jac.append(W_PRIOR * np.eye(m.nv)[self._prior_dof])
        if q_prev is not None:
            dv = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dv, 1.0, q_prev, q)
            rows.append(w_reg * dv[self.rk.indep_dof])
            jac.append(w_reg * np.eye(m.nv)[self.rk.indep_dof])
        return np.concatenate(rows), np.vstack(jac)

    def solve_frame(
        self, q0: np.ndarray, tgt: np.ndarray, contact: list[str], q_prev: np.ndarray | None,
        iters: int = 8, w_contact: float = 2.0, w_pen: float = 5.0, w_reg: float = 0.02,
    ) -> tuple[np.ndarray, float]:  # fmt: skip
        """Projected LM on the independent coordinates; returns ``(qpos, cost)``."""
        m = self.model
        q = self.rk.clamp(q0)
        r, Jf = self._residual(q, tgt, contact, q_prev, w_contact, w_pen, w_reg)
        cost = float(r @ r)
        lam = 1e-3
        for _ in range(iters):
            T = self.rk.velocity_map(q)
            J = Jf @ T
            H = J.T @ J
            g = J.T @ r
            dq = np.linalg.solve(
                H + lam * (np.diag(np.diag(H)) + 1e-6 * np.eye(len(g))), -g
            )
            v = T @ dq
            qn = q.copy()
            mujoco.mj_integratePos(m, qn, v, 1.0)
            qn = self.rk.clamp(qn)
            rn, Jn = self._residual(qn, tgt, contact, q_prev, w_contact, w_pen, w_reg)
            cn = float(rn @ rn)
            if cn < cost:
                q, r, Jf, cost = qn, rn, Jn, cn
                lam = max(lam / 3.0, 1e-6)
                if np.abs(dq).max() < 1e-5:
                    break
            else:
                lam *= 5.0
        return q, cost

    def initial_qpos(self, ref: YogaReference, frame: int = 0) -> np.ndarray:
        """MyoFullBody's rest qpos placed on the reference pelvis (warm start)."""
        q = self.model.qpos0.copy()
        R_pel = quat_to_mat(ref.body_quat[frame, PI["Pelvis"]])
        q_rot = np.zeros(4)
        mujoco.mju_mat2Quat(
            q_rot, (R_pel @ quat_to_mat(self.tpose_qpos[3:7])).reshape(-1)
        )
        q[3:7] = q_rot
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        hip = 0.5 * (
            d.xpos[self.model.body("femur_l").id]
            + d.xpos[self.model.body("femur_r").id]
        )
        tgt = 0.5 * (
            ref.body_pos[frame, PI["L_Hip"]] + ref.body_pos[frame, PI["R_Hip"]]
        )
        q[:3] += tgt - hip
        return q

    def retarget(self, ref: YogaReference, iters: int = 12) -> tuple[np.ndarray, dict]:
        """Solve every frame. Returns ``(qpos (T, nq) at the reference rate, metrics)``."""
        tg = self.targets(ref)
        low = plant_body_lowest(ref, self.colliders)
        contact = low < FLOOR_CONTACT_M
        q = self.initial_qpos(ref)
        q, _ = self.solve_frame(
            q,
            tg[0],
            [PLANT_BODIES[b] for b in np.nonzero(contact[0])[0]],
            None,
            iters=60,
        )
        Q = np.empty((ref.n_frames, self.model.nq))
        prev = None
        costs: list[float] = []
        restarts = 0
        for t in range(ref.n_frames):
            cs = [PLANT_BODIES[b] for b in np.nonzero(contact[t])[0]]
            q, c = self.solve_frame(q, tg[t], cs, prev, iters=iters if t else 30)
            typical = float(np.median(costs[-30:])) if costs else c
            if t and c > max(2.0 * typical, 0.01):
                # Warm starts can stay in a worse basin (e.g. a twisted arm): try a cold start.
                qc, cc = self.solve_frame(
                    self.initial_qpos(ref, t), tg[t], cs, prev, iters=60
                )
                if cc < c:
                    q, c = qc, cc
                    restarts += 1
            costs.append(c)
            Q[t] = q
            prev = q
        met = self.metrics(Q, tg, contact)
        met["cold_restarts"] = restarts
        return Q, met

    def metrics(self, Q: np.ndarray, tg: np.ndarray, contact: np.ndarray) -> dict:
        """Landmark errors, support float, penetration and joint-box saturation."""
        m, d = self.model, self.data
        err = np.empty((len(Q), len(self.landmarks)))
        float_by_body: dict[str, list[float]] = {}
        pen = np.empty(len(Q))
        overlap = np.zeros(len(Q))
        geoms = self.all_colliders
        for t, q in enumerate(Q):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            for i, lm in enumerate(self.landmarks):
                b = self._body_id[lm.myo_body]
                p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self._local[lm.name]
                err[t, i] = np.linalg.norm(p - tg[t, i])
            low = _lowest_points(
                m.geom_type[geoms],
                d.geom_xpos[geoms],
                d.geom_xmat[geoms].reshape(-1, 3, 3),
                m.geom_size[geoms],
            )[:, 2]
            pen[t] = min(0.0, low.min())
            mujoco.mj_collision(m, d)
            overlap[t] = max(
                [0.0]
                + [
                    -c.dist
                    for c in d.contact[: d.ncon]
                    if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]
                ]
            )
            gindex = {g: k for k, g in enumerate(geoms)}
            for b in np.nonzero(contact[t])[0]:
                pb = PLANT_BODIES[b]
                gs = self._contact_geoms.get(pb)
                if gs is None or len(gs) == 0:
                    continue
                float_by_body.setdefault(pb, []).append(
                    float(low[[gindex[g] for g in gs]].min())
                )
        jc = [i for i, lm in enumerate(self.landmarks) if lm.weight >= 1.0]
        q_ind = Q[:, self.rk.indep_hinge_qadr]
        at_lim = np.isclose(q_ind, self.rk.indep_hinge_lo, atol=1e-4) | np.isclose(
            q_ind, self.rk.indep_hinge_hi, atol=1e-4
        )
        names = [
            self.model.joint(int(np.nonzero(self.model.jnt_qposadr == a)[0][0])).name
            for a in self.rk.indep_hinge_qadr
        ]
        floats = (
            np.concatenate([np.asarray(v) for v in float_by_body.values()])
            if float_by_body
            else np.zeros(1)
        )
        return {
            "frames": int(len(Q)),
            "joint_centre_err_m": {"median": float(np.median(err[:, jc])), "p95": float(np.percentile(err[:, jc], 95)), "max": float(err[:, jc].max())},
            "landmark_err_p95_m": {lm.name: float(np.percentile(err[:, i], 95)) for i, lm in enumerate(self.landmarks)},
            "support_float_m": {"median": float(np.median(floats)), "p95": float(np.percentile(floats, 95)), "share_above_2cm": float((floats > 0.02).mean())},
            "support_float_p95_by_body_m": {k: float(np.percentile(v, 95)) for k, v in float_by_body.items()},
            "max_penetration_m": float(-pen.min()),
            "body_overlap_m": {
                "p95": float(np.percentile(overlap, 95)),
                "max": float(overlap.max()),
            },
            "joint_limit_share": {n: float(at_lim[:, k].mean()) for k, n in enumerate(names) if at_lim[:, k].mean() > 0.01},
            "joint_angle_deg": {
                n: [float(v) for v in np.degrees(np.percentile(q_ind[:, k], [0, 0.5, 99.5, 100]))]
                for k, n in enumerate(names)
            },
        }  # fmt: skip


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _support(
    gtype: int, c: np.ndarray, R: np.ndarray, s: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Support point of one primitive along world direction *u*."""
    ul = R.T @ u
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c + s[0] * u
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return c + R @ np.array([0.0, 0.0, np.sign(ul[2]) * s[1]]) + s[0] * u
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        e = s * ul
        return c + R @ (s * e / max(np.linalg.norm(e), 1e-12))
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return c + R @ (np.sign(ul) * s)
    return c.copy()


def _lowest_points(
    gtype: np.ndarray, c: np.ndarray, R: np.ndarray, s: np.ndarray
) -> np.ndarray:
    """``(G, 3)`` lowest world point of each primitive."""
    down = np.array([0.0, 0.0, -1.0])
    return np.array(
        [_support(int(t), ci, Ri, si, down) for t, ci, Ri, si in zip(gtype, c, R, s)]
    )


# --------------------------------------------------------------------------- #
# Clip writing
# --------------------------------------------------------------------------- #
def resample(
    model: mujoco.MjModel,
    rk: ReducedKinematics,
    Q: np.ndarray,
    src_hz: float,
    dst_hz: float,
) -> np.ndarray:
    """Resample a qpos trajectory: hinges linearly, the free-joint quaternion by slerp."""
    T = len(Q)
    t_src = np.arange(T) / src_hz
    n = int(np.floor(t_src[-1] * dst_hz)) + 1
    t_dst = np.arange(n) / dst_hz
    out = np.empty((n, model.nq))
    for k, t in enumerate(t_dst):
        i = min(int(np.floor(t * src_hz)), T - 2)
        a = t * src_hz - i
        q = (1 - a) * Q[i] + a * Q[i + 1]
        quat = np.zeros(4)
        q0, q1 = Q[i, 3:7], Q[i + 1, 3:7]
        if q0 @ q1 < 0:
            q1 = -q1
        dq = np.zeros(3)
        mujoco.mju_subQuat(dq, q1, q0)
        mujoco.mju_copy(quat, q0)
        mujoco.mju_quatIntegrate(quat, dq, a)
        q[3:7] = quat
        out[k] = rk.apply_couplings(q)
    return out


def write_clip(model: mujoco.MjModel, Q: np.ndarray, hz: float, path: Path) -> None:
    """MuscleMimic clip ``.npz``: qpos, qvel (central differences), mimic site positions."""
    T = len(Q)
    V = np.zeros((T, model.nv))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(model, V[t], (b - a) / hz, Q[a], Q[b])
    sites = list(FULLBODY_BODY2SITES_FOR_MIMIC.values())
    sid = [model.site(n).id for n in sites]
    d = mujoco.MjData(model)
    X = np.empty((T, len(sites), 3))
    for t in range(T):
        d.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, d)
        X[t] = d.site_xpos[sid]
    names = [model.joint(j).name for j in range(model.njnt)]
    np.savez_compressed(
        path, qpos=Q.astype(np.float32), qvel=V.astype(np.float32), site_xpos=X.astype(np.float32),
        site_names=np.array(sites), frequency=np.array(hz), joint_order=np.array(names),
    )  # fmt: skip


def _worker(job: tuple) -> tuple[str, dict]:
    table_path, mosh_pkl, smplx, plant_xml, ref_path, out_dir, brace_allowance = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    table = sb.SubjectTable.load(Path(table_path))
    model = sb.compile_spec(sb.build_subject_spec(table))
    reg = registration_model(table) if table.colliders else None
    rt = Retargeter(
        model,
        Path(plant_xml),
        sb.her_template(Path(mosh_pkl), Path(smplx)),
        registration=reg,
        brace_allowance=brace_allowance,
    )
    ref = load_reference(Path(ref_path))
    Q, met = rt.retarget(ref)
    np.save(Path(out_dir) / f"{ref.name}.qpos60.npy", Q)
    Q100 = resample(model, rt.rk, Q, ref.fps, CONTROL_HZ)
    write_clip(model, Q100, CONTROL_HZ, Path(out_dir) / f"{ref.name}.npz")
    (Path(out_dir) / f"{ref.name}.metrics.json").write_text(json.dumps(met, indent=1))
    return ref.name, met


def main() -> None:
    """Retarget references in parallel and print a summary."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--mosh-pkl", type=Path, required=True)
    ap.add_argument("--smplx", type=Path, required=True)
    ap.add_argument("--plant-xml", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--clips", nargs="*", default=None, help="substrings selecting clips"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--brace-allowance",
        type=float,
        default=None,
        help="with a self-colliding table: overlap (m) allowed between primitives",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    refs = sorted(args.refs.glob("*.npz"))
    if args.clips:
        refs = [r for r in refs if any(c in r.stem for c in args.clips)]
    jobs = [
        (
            str(args.table),
            str(args.mosh_pkl),
            str(args.smplx),
            str(args.plant_xml),
            str(r),
            str(args.out),
            args.brace_allowance,
        )
        for r in refs
    ]
    if args.workers <= 1:
        results = [_worker(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
    for name, met in results:
        jc, fl = met["joint_centre_err_m"], met["support_float_m"]
        print(
            f"{name[:60]:60s} jc med {100 * jc['median']:.1f} p95 {100 * jc['p95']:.1f} cm | "
            f"float med {100 * fl['median']:.1f} p95 {100 * fl['p95']:.1f} cm >2cm {fl['share_above_2cm']:.2f} | "
            f"pen {100 * met['max_penetration_m']:.1f} cm"
        )


if __name__ == "__main__":
    main()
