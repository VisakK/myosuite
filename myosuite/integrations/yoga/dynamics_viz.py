# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Deep-dive video of one recorded rollout: muscles, joint torques, contacts, COM/COP.

Reads what ``evaluate --record-state`` saved (``<clip>.state.npz`` and ``model.mjb``),
replays every control step on the CPU (``mj_forward`` on the recorded
``qpos/qvel/act/ctrl`` reproduces the GPU forces) and composes one frame per step:

* left: muscle activations, one row per left/right muscle pair (left bars grow left),
  grouped by the most distal joint region the muscle crosses, on a 0..max-seen scale
  with a tick at each muscle's peak;
* centre: a side render with the floor reaction per hand and foot, the COM and the
  COP; a top-down view of the support polygon with COM, COP and capture point; the
  vertical floor reaction per hand and foot against time;
* right: torque about every independent hinge (joint couplings projected out) against
  time, split into active muscle, passive (muscle + joint) and range-limit parts.

Writes ``<name>.mp4`` (real time, 50 fps), ``<name>_100hz.mp4`` (every step) and, with
``--slowmo``, a 0.25x copy of that span; ``--preview`` writes single frames as PNGs.

CLI (record one replica, then render it)::

    python -m myosuite.integrations.yoga.evaluate --run <run> --checkpoint <ckpt> \
        --refs <ref dir> --plant-xml <plant.xml> --include Handstand --start hold \
        --at-s 12 --horizon-s 10 --repeats 4 --render-best --render Handstand \
        --record-state --out <eval out>
    MUJOCO_GL=egl python -m myosuite.integrations.yoga.dynamics_viz --rollout <eval out> \
        --refs <ref dir> --plant-xml <plant.xml> [--phases kick-up:13.5:14.15] \
        [--slowmo 13.0:15.5] [--preview 150 700]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.yoga.kinematics import ReducedKinematics

GRAVITY = 9.81
CONTACT_GROUPS = ("right hand", "left hand", "right foot", "left foot", "other")
# Display order of the muscle regions, top to bottom.
REGIONS = (
    "Wrist & hand",
    "Elbow & forearm",
    "Shoulder",
    "Neck",
    "Trunk",
    "Hip",
    "Knee",
    "Ankle & foot",
)
# A muscle belongs to the most distal region it crosses.
_DISTAL_FIRST = (
    "Wrist & hand",
    "Elbow & forearm",
    "Shoulder",
    "Ankle & foot",
    "Knee",
    "Hip",
    "Neck",
    "Trunk",
)
# Torque panels: two columns of (joint base name, title).
TORQUE_PANELS = (
    (
        ("flexion", "Wrist flexion"),
        ("deviation", "Wrist deviation"),
        ("pro_sup", "Pronation"),
        ("elbow_flex", "Elbow flexion"),
        ("shoulder_elv", "Shoulder elevation"),
        ("elv_angle", "Shoulder elevation plane"),
        ("shoulder_rot", "Shoulder rotation"),
        ("flex_extension", "Lumbar flexion"),
        ("lat_bending", "Lumbar lateral bending"),
        ("axial_rotation", "Lumbar axial rotation"),
    ),
    (
        ("hip_flexion", "Hip flexion"),
        ("hip_adduction", "Hip adduction"),
        ("hip_rotation", "Hip rotation"),
        ("knee_angle", "Knee angle"),
        ("ankle_angle", "Ankle angle"),
        ("subtalar_angle", "Subtalar angle"),
        ("mtp_angle", "MTP angle"),
        ("neck_flexion", "Neck flexion"),
        ("neck_lateral", "Neck lateral bending"),
        ("neck_rotation", "Neck rotation"),
    ),
)
TORQUE_PARTS = ("active muscle", "passive (muscle + joint)", "range limit")

# Light-mode colours (dataviz reference palette): side identity, COM, neutrals.
RIGHT, LEFT, COM_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def split_name(name: str) -> tuple[str, str | None]:
    """``"prefix/elbow_flex_r"`` -> ``("elbow_flex", "r")``; ``"FCU_left"`` -> ``("FCU", "l")``.

    Names without a side suffix (midline joints, and the right-arm muscles, whose left
    twins end in ``_left``) get ``None``.
    """
    n = name.rsplit("/", 1)[-1]
    if n.endswith("_left"):
        return n[:-5], "l"
    if n[-2:] in ("_r", "_l"):
        return n[:-2], n[-1]
    return n, None


def joint_region(base: str) -> str:
    """Region of a joint from its side-free name."""
    if base in ("flexion", "deviation"):
        return "Wrist & hand"
    if base in ("elbow_flex", "pro_sup"):
        return "Elbow & forearm"
    prefixes = (
        "sternoclavicular",
        "acromioclavicular",
        "unrot",
        "shoulder",
        "elv_angle",
    )
    if base.startswith(prefixes):
        return "Shoulder"
    if base.startswith(("ankle", "subtalar", "mtp")):
        return "Ankle & foot"
    if base.startswith("knee"):
        return "Knee"
    if base.startswith("hip"):
        return "Hip"
    if base.startswith("neck"):
        return "Neck"
    return "Trunk"


def center_of_pressure(
    points: np.ndarray, forces: np.ndarray, z_floor: float = 0.0
) -> np.ndarray:
    """COP on the plane ``z = z_floor`` of point forces (``(n, 3)`` each); NaN if unloaded."""
    points, forces = np.atleast_2d(points), np.atleast_2d(forces)
    fz = forces[:, 2].sum() if len(forces) else 0.0
    if fz <= 1e-6:
        return np.full(2, np.nan)
    r = points - np.array([0.0, 0.0, z_floor])
    moment = np.cross(r, forces).sum(axis=0)
    return np.array([-moment[1], moment[0]]) / fz


@dataclass
class MuscleLayout:
    """Left/right muscle pairs in display order.

    Attributes:
        names: Side-free name per row.
        regions: Region per row.
        right: Muscle index (into the muscle list) of the right muscle per row.
        left: Muscle index of the left muscle per row.
    """

    names: list[str]
    regions: list[str]
    right: np.ndarray
    left: np.ndarray


def muscle_layout(
    model: mujoco.MjModel,
    muscles: np.ndarray,
    qpos_samples: np.ndarray,
    peak: np.ndarray | None = None,
) -> MuscleLayout:
    """Pair the muscles by side and group them by region (``peak`` orders each region).

    Args:
        model: Compiled model.
        muscles: Actuator ids of the muscles.
        qpos_samples: Poses at which moment arms decide which joints a muscle crosses.
        peak: Per-muscle peak activation; rows inside a region go from high to low.
    """
    d = mujoco.MjData(model)
    arm = np.zeros((model.nu, model.nv))
    dense = np.zeros((model.nu, model.nv))
    for q in np.atleast_2d(qpos_samples):
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        mujoco.mju_sparse2dense(
            dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        arm = np.maximum(arm, np.abs(dense))
    dof_region = [
        joint_region(split_name(model.joint(int(model.dof_jntid[v])).name)[0])
        if model.jnt_type[model.dof_jntid[v]] != mujoco.mjtJoint.mjJNT_FREE
        else None
        for v in range(model.nv)
    ]
    by_side: dict[str, dict[str, int]] = {"r": {}, "l": {}}
    region_of: dict[str, str] = {}
    split = [split_name(model.actuator(int(a)).name) for a in muscles]
    left_arm = {b for b, s in split if s == "l"}
    for k, (a, (base, side)) in enumerate(zip(muscles, split)):
        if side is None and base in left_arm:
            side = "r"
        if side is None:
            raise ValueError(f"muscle without a side: {model.actuator(int(a)).name}")
        by_side[side][base] = k
        crossed = {dof_region[v] for v in np.flatnonzero(arm[a] > 1e-4)} - {None}
        region_of[base] = next((r for r in _DISTAL_FIRST if r in crossed), "Trunk")
    if set(by_side["r"]) != set(by_side["l"]):
        raise ValueError("muscles do not pair left/right")
    score = (
        {b: max(peak[by_side["r"][b]], peak[by_side["l"][b]]) for b in by_side["r"]}
        if peak is not None
        else dict.fromkeys(by_side["r"], 0.0)
    )
    rows = sorted(
        by_side["r"], key=lambda b: (REGIONS.index(region_of[b]), -score[b], b)
    )
    return MuscleLayout(
        names=rows,
        regions=[region_of[b] for b in rows],
        right=np.array([by_side["r"][b] for b in rows]),
        left=np.array([by_side["l"][b] for b in rows]),
    )


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in out:
            out.add(b)
    return out


def _joint_body(model: mujoco.MjModel, base: str, side: str) -> int:
    for j in range(model.njnt):
        if split_name(model.joint(j).name) == (base, side):
            return int(model.jnt_bodyid[j])
    raise KeyError(f"{base}_{side}")


def replay(
    model: mujoco.MjModel, state: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-step dynamics of a recorded rollout (see the module docstring).

    Returns arrays over ``T`` steps: ``com``/``com_vel`` (T, 3), ``capture`` and ``cop``
    (T, 2, world xy), ``group_force`` (T, G, 3) and ``group_cop`` (T, G, 2) per
    :data:`CONTACT_GROUPS`, contact ``points``/``point_force``/``point_group`` flattened
    with ``point_count`` (T,), ``torque`` (T, H, 3) about the independent hinges
    ``hinge_names`` split into :data:`TORQUE_PARTS`, ``act`` (T, M) of the ``muscles``,
    and ``eq_leak`` (T,), the largest coupling-constraint torque left after projection.
    """
    rk = ReducedKinematics(model)
    d = mujoco.MjData(model)
    muscles = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    root = int(
        model.jnt_bodyid[
            np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        ]
    )
    group_of_body = np.full(model.nbody, len(CONTACT_GROUPS) - 1)
    for g, (base, side) in enumerate(
        [("flexion", "r"), ("flexion", "l"), ("ankle_angle", "r"), ("ankle_angle", "l")]
    ):
        group_of_body[list(_subtree(model, _joint_body(model, base, side)))] = g
    hinge_cols = {
        int(dof): i
        for i, dof in enumerate(rk.indep_dof)
        if dof in set(rk.indep_hinge_dof)
    }
    hinge_dofs = list(hinge_cols)
    limit_types = (
        mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,
        mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON,
    )
    n = len(state["qpos"])
    out: dict[str, list] = {k: [] for k in (
        "com", "com_vel", "capture", "cop", "group_force", "group_cop", "points",
        "point_force", "point_group", "point_count", "torque", "eq_leak")}  # fmt: skip
    f6 = np.zeros(6)
    for k in range(n):
        d.qpos[:], d.qvel[:] = state["qpos"][k], state["qvel"][k]
        d.act[:], d.ctrl[:] = state["act"][k], state["ctrl"][k]
        mujoco.mj_forward(model, d)
        pts, frc, grp = [], [], []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 == 0) == (b2 == 0):
                continue
            mujoco.mj_contactForce(model, d, i, f6)
            f = c.frame.reshape(3, 3).T @ f6[:3]
            pts.append(c.pos.copy())
            frc.append(f if b1 == 0 else -f)
            grp.append(group_of_body[b2 if b1 == 0 else b1])
        pts_a = np.array(pts).reshape(-1, 3)
        frc_a = np.array(frc).reshape(-1, 3)
        grp_a = np.array(grp, int)
        gf = np.zeros((len(CONTACT_GROUPS), 3))
        gc = np.full((len(CONTACT_GROUPS), 2), np.nan)
        for g in range(len(CONTACT_GROUPS)):
            m = grp_a == g
            if m.any():
                gf[g] = frc_a[m].sum(axis=0)
                gc[g] = center_of_pressure(pts_a[m], frc_a[m])
        com = d.subtree_com[root].copy()
        mujoco.mj_subtreeVel(model, d)
        vel = d.subtree_linvel[root].copy()
        omega = np.sqrt(GRAVITY / max(com[2], 1e-3))
        # Torques: limit and equality rows through J^T, then the passive muscle force
        # (activation and motor controls zeroed) split off the actuator torque.
        lim = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(
            model, d, lim, d.efc_force * np.isin(d.efc_type, limit_types)
        )
        eqf = np.zeros(model.nv)
        eq_rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        mujoco.mj_mulJacTVec(model, d, eqf, d.efc_force * eq_rows)
        total_act, passive = d.qfrc_actuator.copy(), d.qfrc_passive.copy()
        d.act[:], d.ctrl[:] = 0.0, 0.0
        mujoco.mj_fwdActuation(model, d)
        muscle_passive = d.qfrc_actuator.copy()
        T = rk.velocity_map(d.qpos)
        cols = [hinge_cols[v] for v in hinge_dofs]
        parts = np.stack(
            [
                (T.T @ (total_act - muscle_passive))[cols],
                (T.T @ (passive + muscle_passive))[cols],
                (T.T @ lim)[cols],
            ],
            axis=-1,
        )
        out["com"].append(com)
        out["com_vel"].append(vel)
        out["capture"].append(com[:2] + vel[:2] / omega)
        out["cop"].append(center_of_pressure(pts_a, frc_a))
        out["group_force"].append(gf)
        out["group_cop"].append(gc)
        out["points"].append(pts_a[:, :2])
        out["point_force"].append(frc_a)
        out["point_group"].append(grp_a)
        out["point_count"].append(len(pts_a))
        out["torque"].append(parts)
        out["eq_leak"].append(np.abs((T.T @ eqf)[cols]).max(initial=0.0))
    res = {
        k: np.concatenate(v)
        if k in ("points", "point_force", "point_group")
        else np.asarray(v)
        for k, v in out.items()
    }
    res["act"] = np.asarray(state["act"])[:, : len(muscles)]
    res["muscles"] = muscles
    res["hinge_names"] = np.array(
        [
            model.joint(int(model.dof_jntid[v])).name.rsplit("/", 1)[-1]
            for v in hinge_dofs
        ]
    )
    res["weight"] = np.float64(model.body_mass.sum() * GRAVITY)
    res["t"] = float(state["start_s"]) + np.arange(n) / float(state["hz"])
    return res


def _phase_at(t: float, phases: list[tuple[str, float, float]]) -> str:
    return next((p for p, a, b in phases if a <= t < b), "")


class Composer:
    """Builds the figure once and redraws the per-frame artists.

    Args:
        model: The env's compiled model (renders the side view).
        dyn: :func:`replay` output.
        state: The recorded state (for the render).
        ref_draw: ``(scene, t) -> None`` drawing the reference ghost, or ``None``.
        title: Title prefix.
        phases: ``(name, start_s, end_s)`` spans shaded on the time axes.
        size: Frame size in pixels.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        dyn: dict[str, np.ndarray],
        state: dict[str, np.ndarray],
        ref_draw=None,
        title: str = "",
        phases: list[tuple[str, float, float]] | None = None,
        size: tuple[int, int] = (3840, 2160),
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon

        self.model, self.dyn, self.state = model, dyn, state
        self.ref_draw, self.title, self.phases = ref_draw, title, phases or []
        t = dyn["t"]
        self.t = t
        W, H = size
        plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                             "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                             "font.family": "DejaVu Sans"})  # fmt: skip
        self.fig = fig = plt.figure(
            figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE
        )
        self.title_txt = fig.text(
            0.012, 0.98, "", fontsize=20, color=INK, weight="bold", va="center"
        )

        # Side-view frame: hands at the origin, x toward the feet.
        gc = dyn["group_cop"]
        hands = np.nanmean(gc[:, :2], axis=(0, 1))
        feet = np.nanmean(gc[:, 2:4], axis=(0, 1))
        u = feet - hands if np.all(np.isfinite(feet)) else np.array([1.0, 0.0])
        u = u / np.linalg.norm(u)
        self.origin, self.u, self.v = hands, u, np.array([-u[1], u[0]])

        self._muscles(fig)
        self._render_setup(fig, size)
        self._bev(fig, Polygon)
        self._contacts(fig)
        self._torques(fig, Line2D)

    # Coordinates in the top-down frame.
    def _bev_xy(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p)[..., :2] - self.origin
        return np.stack([q @ self.u, q @ self.v], axis=-1)

    def _time_axes(self, ax, names: bool = False) -> None:
        for name, a, b in self.phases:
            for x in (a, b):
                if self.t[0] < x < self.t[-1]:
                    ax.axvline(x, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
            if names:
                ax.text(0.5 * (max(a, self.t[0]) + min(b, self.t[-1])), 1.02, name, ha="center",
                        va="bottom", fontsize=10.5, color=INK2, transform=ax.get_xaxis_transform())  # fmt: skip
        ax.set_xlim(self.t[0], self.t[-1])
        ax.axhline(0.0, color=MUTED, lw=0.6, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _muscles(self, fig) -> None:
        dyn = self.dyn
        act = dyn["act"]
        peak = act.max(axis=0)
        lay = muscle_layout(
            self.model,
            dyn["muscles"],
            self.state["qpos"][:: max(1, len(act) // 6)],
            peak,
        )
        self.layout = lay
        self.xmax = float(np.ceil(act.max() / 0.05) * 0.05)
        gap = 2.4
        y, ycur, prev = [], 0.0, None
        heads = []
        for r in lay.regions:
            if r != prev:
                ycur += gap if prev is not None else 0.0
                heads.append((r, ycur - 1.3))
                prev = r
            y.append(ycur)
            ycur += 1.0
        y = np.array(y)
        top, bot = 0.918, 0.035
        axL = fig.add_axes([0.022, bot, 0.096, top - bot])
        axN = fig.add_axes([0.118, bot, 0.04, top - bot])
        axR = fig.add_axes([0.158, bot, 0.096, top - bot])
        lim = (y.max() + 0.8, -2.2)
        for ax in (axL, axN, axR):
            ax.set_ylim(*lim)
            ax.set_yticks([])
            ax.set_facecolor(SURFACE)
            for s in ("left", "right", "top"):
                ax.spines[s].set_visible(False)
        axN.axis("off")
        axL.set_xlim(self.xmax, 0.0)
        axR.set_xlim(0.0, self.xmax)
        ticks = np.round(np.linspace(0.0, self.xmax, 5), 3)
        for ax, side in ((axL, "left"), (axR, "right")):
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelsize=10, top=True, labeltop=True)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_xlabel(f"activation, {side} side", fontsize=11)
        # Region bands.
        bounds = [h for _, h in heads] + [y.max() + 0.6]
        for i, (name, h) in enumerate(heads):
            for ax in (axL, axN, axR):
                if i % 2 == 0:
                    ax.axhspan(h - 0.6, bounds[i + 1], color="#f1f0ec", lw=0, zorder=0)
            axN.text(0.5, h, name.upper(), ha="center", va="center", fontsize=8.5,
                     weight="bold", color=INK)  # fmt: skip
        for yy, nm in zip(y, lay.names):
            axN.text(0.5, yy, nm, ha="center", va="center", fontsize=6.8, color=INK2)
        self.bars_l = axL.barh(y, act[0, lay.left], height=0.8, color=LEFT, lw=0)
        self.bars_r = axR.barh(y, act[0, lay.right], height=0.8, color=RIGHT, lw=0)
        for ax, idx in ((axL, lay.left), (axR, lay.right)):
            ax.vlines(peak[idx], y - 0.42, y + 0.42, color=INK, lw=0.9, zorder=3)
        fig.text(0.138, 0.955, "Muscle activation  (left ◀ | ▶ right)", ha="center",
                 fontsize=14, weight="bold", color=INK)  # fmt: skip
        fig.text(0.138, 0.942, f"scale 0–{self.xmax:.2f} (highest reached); black tick = each muscle's peak",
                 ha="center", fontsize=10.5, color=INK2)  # fmt: skip

    def _render_setup(self, fig, size) -> None:
        W, H = size
        box = [0.272, 0.425, 0.345, 0.515]
        self.rw, self.rh = int(round(box[2] * W)), int(round(box[3] * H))
        m = self.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.rw)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.rh)
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.75
        m.tendon_width[:] = np.minimum(m.tendon_width, 0.0025)
        m.vis.map.haze = 0.0
        self.renderer = mujoco.Renderer(m, height=self.rh, width=self.rw, max_geom=8000)
        self.data = mujoco.MjData(m)
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        feet = np.nanmean(self.dyn["group_cop"][:, 2:4], axis=(0, 1))
        reach = np.linalg.norm(feet - self.origin) if np.all(np.isfinite(feet)) else 0.0
        mid = self.origin + 0.35 * reach * self.u
        cam.lookat[:] = [mid[0], mid[1], 1.05]
        # Steep enough that the floor (an infinite plane) fills the frame: no skybox. Turned
        # 35 deg off the side view so the two hands (and their arrows) do not overlap.
        cam.distance, cam.elevation = 3.7, -24.0
        cam.azimuth = float(np.degrees(np.arctan2(self.v[1], self.v[0]))) + 35.0
        self.cam = cam
        self.ax_img = fig.add_axes(box)
        self.ax_img.axis("off")
        self.img = self.ax_img.imshow(np.zeros((self.rh, self.rw, 3), np.uint8))
        self.ax_img.set_xlim(0, self.rw)
        self.ax_img.set_ylim(self.rh, 0)
        (self.o_plumb,) = self.ax_img.plot([], [], color=COM_C, lw=2.0, ls=(0, (4, 3)))
        (self.o_com,) = self.ax_img.plot(
            [], [], "o", ms=16, color=COM_C, mec="white", mew=2.0
        )
        (self.o_comg,) = self.ax_img.plot(
            [], [], "o", ms=9, color=COM_C, mec="white", mew=1.5
        )
        (self.o_cop,) = self.ax_img.plot(
            [], [], "X", ms=15, color=INK, mec="white", mew=1.5
        )
        self.o_arrows = [
            self.ax_img.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops={
                "arrowstyle": "-|>,head_length=0.9,head_width=0.45", "lw": 4.5, "color": c,
                "shrinkA": 0, "shrinkB": 0})
            for c in (RIGHT, LEFT, RIGHT, LEFT)
        ]  # fmt: skip
        self.info = self.ax_img.text(0.012, 0.985, "", transform=self.ax_img.transAxes, va="top",
                                     fontsize=12.5, color=INK, family="DejaVu Sans Mono",
                                     bbox={"fc": SURFACE, "ec": GRID, "alpha": 0.85})  # fmt: skip
        self.ax_img.text(0.988, 0.985,
                         "arrows: floor reaction per hand / foot (blue right, orange left; 0.6 m = body weight)\n"
                         "aqua ●: COM and its floor projection   ✕: COP   translucent blue: reference",
                         transform=self.ax_img.transAxes, ha="right", va="top", fontsize=10.5, color=INK2,
                         bbox={"fc": SURFACE, "ec": GRID, "alpha": 0.85})  # fmt: skip

    def _bev(self, fig, Polygon) -> None:
        dyn = self.dyn
        com, cop, cap = (
            self._bev_xy(dyn["com"]),
            self._bev_xy(dyn["cop"]),
            self._bev_xy(dyn["capture"]),
        )
        pts = self._bev_xy(np.c_[dyn["points"], np.zeros(len(dyn["points"]))])
        self.bev_com, self.bev_cop, self.bev_cap = com, cop, cap
        self.bev_ref = self._bev_xy(dyn["ref_com"]) if "ref_com" in dyn else None
        allp = np.concatenate([com, pts, cop[np.isfinite(cop[:, 0])]])
        W, H = self.fig.get_size_inches() * self.fig.dpi
        # Full path, then a zoom on the hands (both equal-aspect; ranges fitted to the box).
        boxes = [[0.285, 0.215, 0.2, 0.18], [0.505, 0.215, 0.105, 0.18]]
        centers = [0.5 * (allp.min(axis=0) + allp.max(axis=0)), np.zeros(2)]
        halves = [
            0.5 * (allp.max(axis=0) - allp.min(axis=0)) + 0.08,
            np.array([0.26, 0.26]),
        ]
        fig.text(0.285, 0.405, "Top-down: support polygon, COM, COP, capture point ξ = x + v/ω",
                 fontsize=13, color=INK, weight="bold")  # fmt: skip
        self.bev = []
        for i, (box, c, half) in enumerate(zip(boxes, centers, halves)):
            ax = fig.add_axes(box)
            aspect = (box[2] * W) / (box[3] * H)
            half = np.maximum(half, [half[1] * aspect, half[0] / aspect])
            ax.set_xlim(c[0] - half[0], c[0] + half[0])
            ax.set_ylim(c[1] - half[1], c[1] + half[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(labelsize=9.5)
            ax.set_xlabel(
                "toward the feet (m), hands at 0" if i == 0 else "hands, zoomed (m)",
                fontsize=11,
            )
            if i == 0:
                ax.set_ylabel("lateral (m)", fontsize=11)
            if self.bev_ref is not None:
                ax.plot(
                    *self.bev_ref.T,
                    color=MUTED,
                    lw=1.0,
                    ls="--",
                    label="reference COM path",
                )
            ax.plot(*com.T, color=COM_C, lw=1.0, alpha=0.35)
            art = {
                "poly": Polygon(
                    np.zeros((3, 2)),
                    closed=True,
                    fc="#e9e8e4",
                    ec=MUTED,
                    lw=1.0,
                    zorder=1,
                )
            }
            ax.add_patch(art["poly"])
            art["pts"] = ax.scatter([], [], s=[], zorder=3, lw=0)
            (art["com_trail"],) = ax.plot([], [], color=COM_C, lw=2.4, zorder=4)
            (art["cop_trail"],) = ax.plot([], [], color=INK, lw=1.4, zorder=4)
            (art["COM"],) = ax.plot(
                [],
                [],
                "o",
                ms=11,
                color=COM_C,
                mec=SURFACE,
                mew=1.5,
                zorder=6,
                label="COM",
            )
            (art["COP"],) = ax.plot(
                [],
                [],
                "X",
                ms=11,
                color=INK,
                mec=SURFACE,
                mew=1.0,
                zorder=6,
                label="COP",
            )
            (art["ξ"],) = ax.plot([], [], "^", ms=10, mfc="none", mec=INK2, mew=1.8, zorder=6,
                                  label="capture point")  # fmt: skip
            (art["ref"],) = ax.plot([], [], "o", ms=10, mfc="none", mec=MUTED, mew=1.5, zorder=5,
                                    label="reference COM")  # fmt: skip
            art["labels"] = {
                k: ax.text(0, 0, k, fontsize=10.5, color=INK, zorder=7)
                for k in ("COM", "COP", "ξ")
            }
            art["span"] = 2 * half[0]
            if i == 0:
                ax.legend(
                    loc="upper right",
                    fontsize=10,
                    frameon=True,
                    framealpha=0.85,
                    ncol=3,
                )
            self.bev.append(art)

    def _contacts(self, fig) -> None:
        dyn = self.dyn
        ax = fig.add_axes([0.285, 0.04, 0.325, 0.125])
        self._time_axes(ax, names=True)
        fz = dyn["group_force"][..., 2] / dyn["weight"]
        styles = [(RIGHT, "-"), (LEFT, "-"), (RIGHT, "--"), (LEFT, "--"), (MUTED, ":")]
        for g, (c, ls) in enumerate(styles):
            if np.abs(fz[:, g]).max() > 0.01:
                ax.plot(
                    self.t, fz[:, g], color=c, ls=ls, lw=1.6, label=CONTACT_GROUPS[g]
                )
        ax.plot(self.t, fz.sum(axis=1), color=INK, lw=0.9, label="total")
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_ylabel("vertical floor\nreaction (BW)", fontsize=11)
        ax.set_xlabel("time in clip (s)", fontsize=11)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.07),
            ncol=6,
            fontsize=10.5,
            frameon=False,
        )
        self.cursors = [ax.axvline(self.t[0], color=INK, lw=1.2)]

    def _torques(self, fig, Line2D) -> None:
        dyn = self.dyn
        names = [str(n) for n in dyn["hinge_names"]]
        tq = dyn["torque"]
        x0s, w, top, bot, n = (0.648, 0.832), 0.163, 0.905, 0.04, 10
        h = (top - bot) / n
        for col, panels in enumerate(TORQUE_PANELS):
            for row, (base, title) in enumerate(panels):
                ax = fig.add_axes([x0s[col], top - (row + 1) * h + 0.01, w, h - 0.026])
                self._time_axes(ax)
                sides = [
                    (s, c)
                    for s, c in (("r", RIGHT), ("l", LEFT))
                    if f"{base}_{s}" in names
                ]
                if not sides and base in names:
                    sides = [(None, INK2)]
                vals = []
                for s, c in sides:
                    j = names.index(base if s is None else f"{base}_{s}")
                    for p, ls, lw in ((0, "-", 1.7), (1, ":", 1.4), (2, "--", 1.2)):
                        ax.plot(self.t, tq[:, j, p], color=c, ls=ls, lw=lw)
                        vals.append(tq[:, j, p])
                if vals:
                    v = np.concatenate(vals)
                    a, b = min(v.min(), 0.0), max(v.max(), 0.0)
                    pad = 0.08 * max(b - a, 1e-3)
                    ax.set_ylim(a - pad, b + pad)
                ax.set_title(
                    title, loc="left", fontsize=11.5, weight="bold", color=INK, pad=3
                )
                ax.tick_params(labelsize=9)
                if row < n - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("time in clip (s)", fontsize=11)
                if col == 0:
                    ax.set_ylabel("N·m", fontsize=10)
                self.cursors.append(ax.axvline(self.t[0], color=INK, lw=1.2))
        handles = [
            Line2D([], [], color=RIGHT, lw=2.5, label="right"),
            Line2D([], [], color=LEFT, lw=2.5, label="left"),
            Line2D([], [], color=INK2, lw=2.5, label="midline"),
            Line2D([], [], color=INK, lw=1.7, ls="-", label=TORQUE_PARTS[0]),
            Line2D([], [], color=INK, lw=1.4, ls=":", label=TORQUE_PARTS[1]),
            Line2D([], [], color=INK, lw=1.2, ls="--", label=TORQUE_PARTS[2]),
        ]
        fig.text(0.648, 0.955, "Joint torques about the independent hinges (positive = joint's + direction)",
                 fontsize=14, weight="bold", color=INK)  # fmt: skip
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.646, 0.932), ncol=6,
                   fontsize=11, frameon=False)  # fmt: skip

    def _project(self, p: np.ndarray) -> np.ndarray:
        """World points ``(n, 3)`` -> render pixel coordinates ``(n, 2)``."""
        sc = self.renderer.scene
        cam = mujoco.mjv_averageCamera(
            sc.camera[0], sc.camera[1]
        )  # mono renders the eye average
        fwd, up = np.asarray(cam.forward), np.asarray(cam.up)
        right = np.cross(fwd, up)
        r = np.atleast_2d(p) - np.asarray(cam.pos)
        z = r @ fwd
        half_h = 0.5 * (cam.frustum_top - cam.frustum_bottom)
        half_w = cam.frustum_width or half_h * self.rw / self.rh
        x = (cam.frustum_near * (r @ right) / z - cam.frustum_center) / half_w
        y = (
            cam.frustum_near * (r @ up) / z
            - 0.5 * (cam.frustum_top + cam.frustum_bottom)
        ) / half_h
        return np.stack([0.5 * self.rw * (1 + x), 0.5 * self.rh * (1 - y)], axis=-1)

    def _overlay(self, k: int) -> None:
        dyn = self.dyn
        weight = float(dyn["weight"])
        for g, ann in enumerate(self.o_arrows):
            f, p = dyn["group_force"][k, g], dyn["group_cop"][k, g]
            show = bool(np.all(np.isfinite(p)) and np.linalg.norm(f) > 0.02 * weight)
            ann.set_visible(show)
            if show:
                a = np.r_[p, 0.0]
                base, tip = self._project(np.stack([a, a + 0.6 * f / weight]))
                ann.xy, ann.xyann = tuple(tip), tuple(base)
        com = dyn["com"][k]
        c, g = self._project(np.stack([com, np.r_[com[:2], 0.0]]))
        self.o_com.set_data([c[0]], [c[1]])
        self.o_comg.set_data([g[0]], [g[1]])
        self.o_plumb.set_data([c[0], g[0]], [c[1], g[1]])
        cop = dyn["cop"][k]
        ok = bool(np.all(np.isfinite(cop)))
        self.o_cop.set_visible(ok)
        if ok:
            q = self._project(np.r_[cop, 0.0])[0]
            self.o_cop.set_data([q[0]], [q[1]])

    def frame(self, k: int) -> np.ndarray:
        """Compose frame ``k``; returns ``(H, W, 3)`` uint8."""
        dyn, st = self.dyn, self.state
        t = float(self.t[k])
        d = self.data
        d.qpos[:], d.qvel[:], d.act[:] = st["qpos"][k], st["qvel"][k], st["act"][k]
        mujoco.mj_forward(self.model, d)
        self.renderer.update_scene(d, camera=self.cam, scene_option=self.opt)
        if self.ref_draw is not None:
            n0 = self.renderer.scene.ngeom
            self.ref_draw(self.renderer.scene, t)
            for g in range(n0, self.renderer.scene.ngeom):
                self.renderer.scene.geoms[g].rgba[3] *= 0.6
        img = self.renderer.render()
        img[img.max(axis=-1) < 3] = (252, 252, 251)  # no skybox: light background
        self.img.set_data(img)
        self._overlay(k)
        act = dyn["act"][k]
        for b, a in zip(self.bars_l, act[self.layout.left]):
            b.set_width(a)
        for b, a in zip(self.bars_r, act[self.layout.right]):
            b.set_width(a)
        for c in self.cursors:
            c.set_xdata([t, t])
        # Top-down views.
        lo = max(0, k - 50)
        c0 = int(dyn["point_count"][:k].sum())
        c1 = c0 + int(dyn["point_count"][k])
        pts = self._bev_xy(np.c_[dyn["points"][c0:c1], np.zeros(c1 - c0)])
        fz = np.maximum(dyn["point_force"][c0:c1, 2], 0.0)
        grp = dyn["point_group"][c0:c1]
        colors = np.where(grp % 2 == 0, RIGHT, LEFT)
        colors[grp == len(CONTACT_GROUPS) - 1] = MUTED
        loaded = pts[fz > 1.0]
        hull = None
        if len(loaded) >= 3:
            from scipy.spatial import ConvexHull, QhullError

            try:
                hull = loaded[ConvexHull(loaded).vertices]
            except QhullError:
                hull = None
        marks = {"COM": self.bev_com[k], "COP": self.bev_cop[k], "ξ": self.bev_cap[k]}
        # Label offsets as fractions of the view width: COM up-right, COP down-right, ξ up-left.
        offsets = {"COM": (0.02, 0.05), "COP": (0.02, -0.09), "ξ": (-0.06, 0.05)}
        for art in self.bev:
            art["com_trail"].set_data(*self.bev_com[lo : k + 1].T)
            art["cop_trail"].set_data(*self.bev_cop[lo : k + 1].T)
            for key, p in marks.items():
                ok = bool(np.all(np.isfinite(p)))
                art[key].set_data([p[0]] if ok else [], [p[1]] if ok else [])
                art["labels"][key].set_visible(ok)
                if ok:
                    dx, dy = offsets[key]
                    art["labels"][key].set_position(
                        (p[0] + dx * art["span"], p[1] + dy * art["span"])
                    )
            if self.bev_ref is not None:
                art["ref"].set_data([self.bev_ref[k, 0]], [self.bev_ref[k, 1]])
            art["pts"].set_offsets(pts if len(pts) else np.zeros((0, 2)))
            art["pts"].set_sizes(20 + 400 * fz / float(dyn["weight"]))
            art["pts"].set_color(colors)
            art["poly"].set_visible(hull is not None)
            if hull is not None:
                art["poly"].set_xy(hull)
        # Text.
        com, vel = dyn["com"][k], dyn["com_vel"][k]
        cop = dyn["cop"][k]
        off = (
            np.linalg.norm(com[:2] - cop) * 100 if np.all(np.isfinite(cop)) else np.nan
        )
        fz_tot = dyn["group_force"][k, :, 2].sum() / float(dyn["weight"])
        self.info.set_text(
            f"t = {t:6.2f} s\n"
            f"COM height     {com[2]:5.2f} m\n"
            f"COM speed (xy) {np.linalg.norm(vel[:2]):5.2f} m/s\n"
            f"COM–COP (xy)   {off:5.1f} cm\n"
            f"floor reaction {fz_tot:5.2f} BW"
        )
        phase = _phase_at(t, self.phases)
        self.title_txt.set_text(
            f"{self.title}   ·   t = {t:5.2f} s" + (f"   ·   {phase}" if phase else "")
        )
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def _add_connector(
    scene: mujoco.MjvScene,
    a: np.ndarray,
    b: np.ndarray,
    width: float,
    rgba: np.ndarray,
    gtype: int = mujoco.mjtGeom.mjGEOM_ARROW,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.objtype, g.objid, g.segid, g.transparent, g.camdist = 0, -1, -1, 1, 0.0
    scene.ngeom += 1


# Worker state of :func:`_render_chunk` (one composer per process).
_W: dict = {}


def _worker_init(args: dict) -> None:
    _W["composer"] = _build_composer(**args)


def _render_chunk(job: tuple[int, int, str]) -> str:
    import imageio.v2 as imageio

    a, b, path = job
    comp = _W["composer"]
    with imageio.get_writer(path, fps=100, codec="libx264", quality=None, macro_block_size=8,
                            pixelformat="yuv420p", ffmpeg_params=["-crf", "16", "-preset", "fast"]) as w:  # fmt: skip
        for k in range(a, b):
            w.append_data(comp.frame(k))
    return path


def _build_composer(
    rollout: str, clip: str, refs: str | None, plant_xml: str | None, title: str,
    phases: list[tuple[str, float, float]],
) -> Composer:  # fmt: skip
    rollout_p = Path(rollout)
    model = mujoco.MjModel.from_binary_path(str(rollout_p / "model.mjb"))
    state = dict(np.load(rollout_p / f"{clip}.state.npz"))
    dyn = dict(np.load(rollout_p / f"{clip}.dynamics.npz"))
    ref_draw = None
    if refs and plant_xml:
        from myosuite.integrations.yoga.reference_io import (
            load_reference,
            plant_body_lowest,
            plant_colliders,
        )
        from myosuite.integrations.yoga.render import draw_reference

        ref = load_reference(Path(refs) / f"{clip}.npz")
        cols = plant_colliders(Path(plant_xml))
        contact = plant_body_lowest(ref, cols) < 0.015

        def ref_draw(scene: mujoco.MjvScene, t: float) -> None:
            j = min(int(round(t * ref.fps)), ref.n_frames - 1)
            draw_reference(scene, ref, j, cols, contact[j])

    return Composer(model, dyn, state, ref_draw, title, phases)


def reference_com(
    model: mujoco.MjModel, clip_qpos: np.ndarray, frames: np.ndarray
) -> np.ndarray:
    """COM of the root subtree for reference poses ``clip_qpos[frames]``."""
    d = mujoco.MjData(model)
    root = int(
        model.jnt_bodyid[
            np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        ]
    )
    out = []
    for f in frames:
        d.qpos[:] = clip_qpos[min(int(f), len(clip_qpos) - 1)]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        out.append(d.subtree_com[root].copy())
    return np.asarray(out)


def ensure_dynamics(rollout: Path, clip: str) -> dict[str, np.ndarray]:
    """:func:`replay` of ``<clip>.state.npz`` plus the reference COM, cached next to it."""
    path = rollout / f"{clip}.dynamics.npz"
    if path.exists():
        return dict(np.load(path))
    model = mujoco.MjModel.from_binary_path(str(rollout / "model.mjb"))
    dyn = replay(model, dict(np.load(rollout / f"{clip}.state.npz")))
    ckpt = Path(json.loads((rollout / "metrics.json").read_text())["checkpoint"])
    cfg = json.loads((ckpt.parent / "config.json").read_text())["args"]
    clip_npz = np.load(Path(cfg["clips_dir"]) / f"{clip}.npz")
    frames = np.round(dyn["t"] * float(clip_npz["frequency"])).astype(int)
    dyn["ref_com"] = reference_com(model, clip_npz["qpos"].astype(float), frames)
    np.savez(path, **dyn)
    fz = dyn["group_force"][..., 2].sum(axis=1) / dyn["weight"]
    check = {"steps": len(dyn["t"]), "eq_leak_max_Nm": float(dyn["eq_leak"].max()),
             "floor_reaction_BW_mean": float(fz.mean()), "act_max": float(dyn["act"].max()),
             "weight_N": float(dyn["weight"])}  # fmt: skip
    (rollout / f"{clip}.dynamics.json").write_text(json.dumps(check, indent=1))
    print("replay:", check)
    return dyn


def main() -> None:
    """Replay a recorded rollout and render the deep-dive video."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rollout", type=Path, required=True, help="evaluate --record-state output dir"
    )
    ap.add_argument(
        "--clip", default=None, help="clip name (default: the only *.state.npz)"
    )
    ap.add_argument("--refs", type=Path, default=None, help="reference dir (ghost)")
    ap.add_argument("--plant-xml", type=Path, default=None)
    ap.add_argument(
        "--phases", nargs="*", default=[], help="name:start_s:end_s, shaded and named"
    )
    ap.add_argument(
        "--slowmo", default=None, help="start_s:end_s (clip time) for a 0.25x copy"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--preview",
        type=int,
        nargs="*",
        default=None,
        help="write these steps as PNGs only",
    )
    ap.add_argument("--name", default="dynamics")
    args = ap.parse_args()
    clip = args.clip or next(args.rollout.glob("*.state.npz")).name.removesuffix(
        ".state.npz"
    )
    dyn = ensure_dynamics(args.rollout, clip)
    ckpt = Path(json.loads((args.rollout / "metrics.json").read_text())["checkpoint"])
    phases = [
        (p.split(":")[0], float(p.split(":")[1]), float(p.split(":")[2]))
        for p in args.phases
    ]
    title = f"{ckpt.parent.name} {ckpt.stem}  ·  {clip.split('_')[1]} from {float(dyn['t'][0]):.1f} s"
    init = {"rollout": str(args.rollout), "clip": clip, "refs": str(args.refs) if args.refs else None,
            "plant_xml": str(args.plant_xml) if args.plant_xml else None, "title": title, "phases": phases}  # fmt: skip
    if args.preview is not None:
        from PIL import Image

        comp = _build_composer(**init)
        for k in args.preview:
            out = args.rollout / f"{args.name}_preview_{k:04d}.png"
            Image.fromarray(comp.frame(k)).save(out)
            print("wrote", out)
        return
    n = len(dyn["t"])
    workers = max(1, min(args.workers, n // 20))
    edges = np.linspace(0, n, workers + 1).astype(int)
    seg_dir = args.rollout / f"{args.name}_segments"
    seg_dir.mkdir(exist_ok=True)
    jobs = [
        (int(edges[i]), int(edges[i + 1]), str(seg_dir / f"seg_{i:02d}.mp4"))
        for i in range(workers)
    ]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(
        workers, initializer=_worker_init, initargs=(init,)
    ) as pool:
        paths = pool.map(_render_chunk, jobs, chunksize=1)
    listing = seg_dir / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in paths))
    full = args.rollout / f"{args.name}_100hz.mp4"
    ff = ["ffmpeg", "-y", "-loglevel", "error"]
    subprocess.run(
        [
            *ff,
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(full),
        ],
        check=True,
    )
    enc = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    real = args.rollout / f"{args.name}.mp4"
    subprocess.run([*ff, "-i", str(full), "-vf", "fps=50", *enc, str(real)], check=True)
    print("wrote", real)
    if args.slowmo:
        lo, hi = (float(x) - float(dyn["t"][0]) for x in args.slowmo.split(":"))
        slow = args.rollout / f"{args.name}_slowmo_0.25x.mp4"
        subprocess.run([*ff, "-ss", f"{max(lo, 0.0):.2f}", "-to", f"{hi:.2f}", "-i", str(full),
                        "-vf", "setpts=4.0*PTS,fps=25", *enc, str(slow)], check=True)  # fmt: skip
        print("wrote", slow)


if __name__ == "__main__":
    main()
