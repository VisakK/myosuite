# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle coordination layer: a joint-space action space for the muscle body, on the GPU.

The muscle-level tracker (one excitation per muscle) learnt the multi-support holds and
none of the balance-critical ones (README §5). The diagnosis: a PD plant gets posture
impedance from its servos, while a muscle policy has to synthesise it from 354 redundant
muscles behind 10-40 ms activation dynamics, and Gaussian noise on excitations is itself
co-contraction. This module gives the policy the PD teacher's action space instead and
leaves the muscle redundancy to a fixed layer, run at every physics substep for every
environment in one batch:

1. PD law on the independent hinges (:mod:`.kinematics`): with ``a_q`` the policy's target
   offsets from the reference one control step ahead,
   ``tau_d = Kp (clip(q_ref + s a_q, lo, hi) - q) + Kd (qd_ref - qd)``.
2. The three neck torque motors (MyoFullBody has no neck muscles): ``ctrl = tau_d / gear``.
3. Muscle generalised forces per unit activation on the independent hinges:
   ``A = T(q)^T R(q)^T diag(gain(l, v))`` and ``b = T(q)^T R(q)^T bias(l)``, with ``R`` the
   tendon Jacobian (the muscles' moment arms), ``T`` the joint-coupling velocity map, and
   ``gain`` / ``bias`` MuJoCo's muscle force-length-velocity and passive curves.
4. Static optimisation about a co-contraction level ``c`` the policy sets per limb:
   ``a* = argmin |a - c|^2 + rho |S^-1 (A a + b - tau_d)|^2`` with ``0 <= a <= 1``, solved
   on the free set in closed form (batched Cholesky) with a few active-set passes.
5. Inverse activation dynamics: the excitation that brings the activation to ``a*`` over
   ``horizon_s`` under MuJoCo's muscle activation model.

The muscles still produce every force, with their strength limits, passive tissue and
activation dynamics; the layer has no learned parameters.

The muscle model, moment-arm gather and static optimisation are plain torch functions so the
CPU tests can check them against MuJoCo.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import mujoco
import numpy as np

from myosuite.integrations.yoga import subject_body as sb
from myosuite.integrations.yoga.kinematics import ReducedKinematics

# PD gains per independent hinge (N m / rad), from the PhysX teacher's plant (ProtoMotions
# ``smpl_yogi_v2``: hips/knees/ankles 800, toes 500, trunk 1000, shoulders/elbows 500,
# wrists 300). Kd = kd_ratio * Kp, the teacher's ratio. The neck uses its own gains: its
# motors saturate at 30 N m.
HINGE_GAINS: tuple[tuple[str, float], ...] = (
    (r"^(flex_extension|lat_bending|axial_rotation)$", 1000.0),
    (r"^(elv_angle|shoulder_elv|shoulder_rot|elbow_flex)_[lr]$", 500.0),
    (r"^(pro_sup|deviation|flexion)_[lr]$", 300.0),
    (
        r"^(hip_flexion|hip_adduction|hip_rotation|knee_angle|ankle_angle|subtalar_angle)_[lr]$",
        800.0,
    ),
    (r"^mtp_angle_[lr]$", 500.0),
)
# Co-contraction groups (one policy action each): which hinges belong to which limb.
COCON_GROUPS: tuple[tuple[str, str], ...] = (
    ("trunk", r"^(flex_extension|lat_bending|axial_rotation)$"),
    ("arm_r", r"_r$:arm"),
    ("arm_l", r"_l$:arm"),
    ("leg_r", r"_r$:leg"),
    ("leg_l", r"_l$:leg"),
)
_ARM = (
    r"^(elv_angle|shoulder_elv|shoulder_rot|elbow_flex|pro_sup|deviation|flexion)_[lr]$"
)
_LEG = r"^(hip_flexion|hip_adduction|hip_rotation|knee_angle|ankle_angle|subtalar_angle|mtp_angle)_[lr]$"


@dataclasses.dataclass(frozen=True)
class CoordinationCfg:
    """Settings of the muscle coordination layer.

    Attributes:
        dq_scale: Target offset per unit action (rad), added to the reference.
        gain_mode: ``"bounded"`` (default): the PhysX plant's gains (:data:`HINGE_GAINS`
            times ``kp_scale``), capped by the muscles' capacity (``sat_angle``) and by what
            the activation lag can stabilise for the hinge's joint-space inertia ``I``
            (``Kp <= I omega_max^2``), with ``Kd = 2 zeta sqrt(Kp I)``. ``"inertia"``:
            computed-torque gains ``I (omega^2 e + 2 zeta omega e')`` (too soft on light
            hinges: the raised leg of a one-leg dog spins about its axis). ``"teacher"``: the
            plant's gains unchanged (``Kd = kd_ratio Kp``); they ring behind the activation
            lag on light hinges (measured).
        omega: Natural frequency (rad/s) of the ``"inertia"`` mode.
        omega_max: Highest natural frequency the ``"bounded"`` mode allows (rad/s).
        zeta: Damping ratio (``"inertia"``, ``"bounded"``).
        accel_ff: Add ``I * qacc_ref`` (the reference's acceleration) to the demand.
        tau_res_scale: Above 0, the action also carries a torque residual per hinge,
            ``tau_res_scale * capacity * a`` (neck: times the motor gear): direct torque
            authority (balance) that does not depend on the PD gains.
        kp_scale: Multiplies every hinge gain of :data:`HINGE_GAINS` (``"teacher"`` mode).
        kd_ratio: ``Kd = kd_ratio * Kp`` (``"teacher"`` mode).
        neck_kp: Neck stiffness (N m / rad); the motors saturate at their gear (30 N m).
        neck_kd: Neck damping (N m s / rad).
        cocon_base: Co-contraction level at zero action.
        cocon_scale: Co-contraction change per unit action.
        cocon_max: Upper clip of the co-contraction level.
        rho: Weight of the normalised torque residual in the static optimisation.
        active_set_iters: Active-set passes of the static optimisation.
        horizon_s: The excitation brings the activation to ``a*`` over this horizon.
        every_substep: Re-solve at every physics substep (else once per control step).
        residual_scale: Above 0, a per-muscle excitation residual is added to the action
            (``n_muscles`` more action dimensions), scaled by this.
        target_lead_steps: PD targets are the reference this many control steps ahead.
        min_capacity: Hinges whose muscle torque capacity is below this (N m) are left out
            of the static optimisation.
        sat_angle: Hinge stiffness is capped at ``capacity / sat_angle``: a weak joint (the
            wrist's muscles give ~9 N m) saturates at this error (rad) instead of switching
            bang-bang between its flexors and extensors at the teacher's gains.
        ff_scale: Multiplies the static feed-forward torques (:mod:`.feedforward`); 0 turns
            them off.
        ff_limits: Feed-forward LP limit policy: ``"all"`` (every hinge at a range limit
            takes a free, rigid reaction; round 2) or ``"stiff"`` (only joints with a stiff
            end-range in the subject table). See :func:`.feedforward.clip_feedforward`.
        planted_inertia: Use the contact-aware hinge inertia (a hinge whose distal segment
            is on the floor drives the rest of the body) for the gain cap and the
            acceleration feed-forward, instead of the free segment's. Measured worse at zero
            action on the round-3 body (frozen handstand 1.80 -> 1.35 s, start sweep 1.26 ->
            0.68 s): the large damping and acceleration terms amplify velocity noise.
    """

    dq_scale: float = 0.4
    gain_mode: str = "bounded"
    omega: float = 15.0
    omega_max: float = 40.0
    zeta: float = 1.0
    accel_ff: bool = True
    kp_scale: float = 1.0
    kd_ratio: float = 0.1
    neck_kp: float = 100.0
    neck_kd: float = 10.0
    cocon_base: float = 0.03
    cocon_scale: float = 0.1
    cocon_max: float = 0.4
    rho: float = 1.0e4
    active_set_iters: int = 4
    horizon_s: float = 0.01
    every_substep: bool = True
    residual_scale: float = 0.0
    target_lead_steps: int = 1
    min_capacity: float = 1.0
    sat_angle: float = 0.2
    ff_scale: float = 1.0
    tau_res_scale: float = 0.0
    ff_limits: str = "all"
    planted_inertia: bool = False


# ---------------------------------------------------------------------------
# Model tables (numpy, from a compiled MjModel)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CoordinationTables:
    """Everything the layer needs from the model, as numpy arrays.

    Rows of the static optimisation are the muscle-driven independent hinges; the neck
    hinges are driven by their motors. ``slot_*`` describe the padded per-muscle rows of
    the sparse tendon Jacobian (``K`` slots per muscle).
    """

    hinge_names: list[str]
    hinge_qadr: np.ndarray  # (H,)
    hinge_dof: np.ndarray  # (H,)
    hinge_lo: np.ndarray
    hinge_hi: np.ndarray
    kp: np.ndarray  # (H,)
    kd: np.ndarray
    muscle_rows: np.ndarray  # (R,) indices into the H hinges
    neck_rows: np.ndarray  # (3,) indices into the H hinges
    neck_actuators: np.ndarray  # (3,) actuator ids
    neck_gear: np.ndarray  # (3,)
    muscle_actuators: np.ndarray  # (M,) actuator ids
    muscle_tendons: np.ndarray  # (M,) tendon ids
    gainprm: np.ndarray  # (M, 10)
    biasprm: np.ndarray  # (M, 10)
    lengthrange: np.ndarray  # (M, 2)
    acc0: np.ndarray  # (M,)
    tau_act: np.ndarray  # (M,)
    tau_deact: np.ndarray  # (M,)
    coupling_ind_qadr: np.ndarray  # (C,)
    coupling_x0: np.ndarray  # (C,)
    coupling_coef: np.ndarray  # (C, 5)
    slot_adr: np.ndarray  # (M, K) index into ten_J (0 for padding)
    slot_row: np.ndarray  # (M, K) static-optimisation row, R for "no row"
    slot_fac: (
        np.ndarray
    )  # (M, K) 0: identity, 1 + j: coupling j, C + 1: padding (factor 0)
    capacity: np.ndarray  # (R,) torque scale per row (N m)
    active_rows: np.ndarray  # (R,) bool: capacity >= min_capacity
    muscle_group: np.ndarray  # (M,) co-contraction group index
    group_names: list[str]


def _hinge_gain(name: str) -> float:
    for pattern, gain in HINGE_GAINS:
        if re.match(pattern, name):
            return gain
    raise KeyError(f"no PD gain for hinge {name!r}")


def _hinge_group(name: str) -> int:
    if re.match(COCON_GROUPS[0][1], name):
        return 0
    side = name.rsplit("_", 1)[-1]
    if re.match(_ARM, name):
        return 1 if side == "r" else 2
    if re.match(_LEG, name):
        return 3 if side == "r" else 4
    raise KeyError(f"hinge {name!r} has no co-contraction group")


def build_tables(
    model: mujoco.MjModel,
    cfg: CoordinationCfg,
    frames_qpos: np.ndarray | None = None,
) -> CoordinationTables:
    """Model tables of the coordination layer.

    Args:
        model: Compiled subject model (muscles first, then the neck motors).
        cfg: Layer settings (gains, minimum capacity).
        frames_qpos: ``(F, nq)`` poses over which the torque capacity of each hinge and the
            co-contraction group of each muscle are measured (default: the model's qpos0).

    Returns:
        The tables.
    """
    rk = ReducedKinematics(model)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[d]))
        for d in rk.indep_hinge_dof
    ]
    neck = [names.index(j) for j in sb.NECK_JOINTS]
    rows = [i for i in range(len(names)) if i not in neck]
    kp = np.array(
        [
            cfg.neck_kp if i in neck else cfg.kp_scale * _hinge_gain(n)
            for i, n in enumerate(names)
        ]
    )
    kd = np.array(
        [
            cfg.neck_kd if i in neck else cfg.kd_ratio * k
            for i, k in zip(range(len(names)), kp)
        ]
    )
    muscles = np.array(
        [
            i
            for i in range(model.nu)
            if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
        ]
    )
    if muscles.size == 0 or (np.diff(muscles) != 1).any() or muscles[0] != 0:
        raise ValueError("expected the muscles to be the first actuators")
    if (model.actuator_trntype[muscles] != mujoco.mjtTrn.mjTRN_TENDON).any():
        raise ValueError("every muscle must pull a tendon")
    if not np.allclose(model.actuator_gear[muscles, 0], 1.0):
        raise ValueError("muscle gears other than 1 are not supported")
    neck_act = np.array([model.actuator(sb.NECK_ACTUATORS[k]).id for k in range(3)])
    tendons = model.actuator_trnid[muscles, 0].astype(np.int64)
    # Couplings: dependent dof -> (independent dof, coefficients).
    dep_to_coupling = {}
    for j, c in enumerate(rk.couplings):
        dep_to_coupling[c.dep_dof] = j
    hinge_row = {int(d): i for i, d in enumerate(rk.indep_hinge_dof)}
    muscle_row = {int(rk.indep_hinge_dof[h]): r for r, h in enumerate(rows)}
    n_c = len(rk.couplings)
    # Padded slots of the sparse tendon Jacobian, one row per muscle.
    rownnz = model.ten_J_rownnz[tendons].astype(np.int64)
    k_max = int(rownnz.max())
    slot_adr = np.zeros((len(muscles), k_max), np.int64)
    slot_row = np.full((len(muscles), k_max), len(rows), np.int64)
    slot_fac = np.full((len(muscles), k_max), n_c + 1, np.int64)
    for i, t in enumerate(tendons):
        adr = int(model.ten_J_rowadr[t])
        for k in range(int(rownnz[i])):
            dof = int(model.ten_J_colind[adr + k])
            slot_adr[i, k] = adr + k
            if dof in muscle_row:
                slot_row[i, k], slot_fac[i, k] = muscle_row[dof], 0
            elif dof in dep_to_coupling:
                c = rk.couplings[dep_to_coupling[dof]]
                if c.ind_dof in muscle_row:
                    slot_row[i, k] = muscle_row[c.ind_dof]
                    slot_fac[i, k] = 1 + dep_to_coupling[dof]
            # else: root, neck, locked or coupled to a non-muscle hinge: no row (factor 0)
    del hinge_row
    tables = CoordinationTables(
        hinge_names=names, hinge_qadr=rk.indep_hinge_qadr.copy(), hinge_dof=rk.indep_hinge_dof.copy(),
        hinge_lo=rk.indep_hinge_lo.copy(), hinge_hi=rk.indep_hinge_hi.copy(), kp=kp, kd=kd,
        muscle_rows=np.array(rows), neck_rows=np.array(neck), neck_actuators=neck_act,
        neck_gear=model.actuator_gear[neck_act, 0].copy(), muscle_actuators=muscles, muscle_tendons=tendons,
        gainprm=model.actuator_gainprm[muscles].copy(), biasprm=model.actuator_biasprm[muscles].copy(),
        lengthrange=model.actuator_lengthrange[muscles].copy(), acc0=model.actuator_acc0[muscles].copy(),
        tau_act=model.actuator_dynprm[muscles, 0].copy(), tau_deact=model.actuator_dynprm[muscles, 1].copy(),
        coupling_ind_qadr=np.array([c.ind_qadr for c in rk.couplings]),
        coupling_x0=np.array([c.x0 for c in rk.couplings]),
        coupling_coef=np.array([c.coef for c in rk.couplings]),
        slot_adr=slot_adr, slot_row=slot_row, slot_fac=slot_fac,
        capacity=np.ones(len(rows)), active_rows=np.ones(len(rows), bool),
        muscle_group=np.zeros(len(muscles), np.int64), group_names=[g for g, _ in COCON_GROUPS],
    )  # fmt: skip
    _measure_capacity(model, tables, cfg, frames_qpos)
    return tables


def _measure_capacity(
    model: mujoco.MjModel,
    tables: CoordinationTables,
    cfg: CoordinationCfg,
    frames_qpos: np.ndarray | None,
) -> None:
    """Torque capacity per row and co-contraction group per muscle, over *frames_qpos*.

    Capacity of a row is the larger of the hinge's maximal flexing and extending muscle
    torques (all its agonists fully active, isometric), median over the frames. A muscle
    joins the group of the row it can torque most.
    """
    import torch

    poses = np.asarray(frames_qpos if frames_qpos is not None else model.qpos0[None])
    data = mujoco.MjData(model)
    t = torch_tables(tables, "cpu", torch.float64)
    caps, shares = [], np.zeros((len(tables.muscle_actuators), len(tables.muscle_rows)))
    for q in poses:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        ten_j = torch.as_tensor(data.ten_J[None])
        qpos = torch.as_tensor(data.qpos[None])
        length = torch.as_tensor(data.actuator_length[None, tables.muscle_actuators])
        gain, _ = muscle_gain_bias(length, torch.zeros_like(length), t)
        a_mat, _ = muscle_matrix(ten_j, qpos, gain, torch.zeros_like(gain), t)
        a = a_mat[0].numpy()  # (R, M)
        caps.append(np.maximum(np.clip(a, 0, None).sum(1), np.clip(-a, 0, None).sum(1)))
        shares += np.abs(a).T
    tables.capacity = np.median(np.array(caps), axis=0)
    tables.active_rows = tables.capacity >= cfg.min_capacity
    rows = tables.muscle_rows
    cap_gain = tables.capacity / max(cfg.sat_angle, 1e-6)
    tables.kp[rows] = np.minimum(tables.kp[rows], cap_gain)
    tables.kd[rows] = cfg.kd_ratio * tables.kp[rows]
    row_group = np.array(
        [_hinge_group(tables.hinge_names[h]) for h in tables.muscle_rows]
    )
    tables.muscle_group = row_group[np.argmax(shares, axis=1)]


def sample_frames(clips: tuple[Any, ...], n: int) -> np.ndarray:
    """``(n, nq)`` poses evenly spaced over the concatenated frames of *clips*."""
    qpos = np.concatenate([np.asarray(c.qpos) for c in clips], axis=0)
    idx = np.linspace(0, len(qpos) - 1, min(n, len(qpos))).round().astype(int)
    return qpos[idx]


# ---------------------------------------------------------------------------
# Torch functions (batched over environments; device-agnostic)
# ---------------------------------------------------------------------------


def torch_tables(
    tables: CoordinationTables, device: Any, dtype: Any = None
) -> dict[str, Any]:
    """The numeric tables as torch tensors on *device* (floats in *dtype*, default float32)."""
    import torch

    dtype = dtype or torch.float32
    out: dict[str, Any] = {}
    for f in dataclasses.fields(tables):
        v = getattr(tables, f.name)
        if isinstance(v, np.ndarray):
            kind = (
                torch.bool
                if v.dtype == bool
                else (torch.long if v.dtype.kind in "iu" else dtype)
            )
            out[f.name] = torch.as_tensor(v, dtype=kind, device=device)
    out["n_couplings"] = len(tables.coupling_x0)
    return out


def _fl_curve(x: Any, lmin: Any, lmax: Any) -> Any:
    """MuJoCo's muscle force-length curve (``mju_muscleGainLength``)."""
    import torch

    a = 0.5 * (lmin + 1.0)
    b = 0.5 * (1.0 + lmax)
    y1 = 0.5 * ((x - lmin) / (a - lmin)) ** 2
    y2 = 1.0 - 0.5 * ((1.0 - x) / (1.0 - a)) ** 2
    y3 = 1.0 - 0.5 * ((x - 1.0) / (b - 1.0)) ** 2
    y4 = 0.5 * ((lmax - x) / (lmax - b)) ** 2
    out = torch.where(
        x <= a, y1, torch.where(x <= 1.0, y2, torch.where(x <= b, y3, y4))
    )
    return torch.where((x < lmin) | (x > lmax), torch.zeros_like(x), out)


def muscle_gain_bias(length: Any, velocity: Any, t: dict[str, Any]) -> tuple[Any, Any]:
    """MuJoCo's muscle force per unit activation and passive force, ``(N, M)`` each.

    Both are negative (pulling), as in ``actuator_force = gain * act + bias``. The active
    part reads ``gainprm``, the passive part ``biasprm`` (they differ where the passive
    curve was fitted separately).
    """
    import torch

    gp, bp, lr = t["gainprm"], t["biasprm"], t["lengthrange"]

    def force_scale(prm: Any) -> Any:
        f = prm[:, 2]
        return torch.where(f < 0, prm[:, 3] / t["acc0"].clamp(min=1e-10), f)

    l0g = (lr[:, 1] - lr[:, 0]) / (gp[:, 1] - gp[:, 0])
    lg = gp[:, 0] + (length - lr[:, 0]) / l0g
    vg = velocity / (l0g * gp[:, 6])
    fvmax = gp[:, 8]
    y = fvmax - 1.0
    fv = torch.where(
        vg <= -1.0,
        torch.zeros_like(vg),
        torch.where(
            vg <= 0.0,
            (vg + 1.0) ** 2,
            torch.where(vg <= y, fvmax - (y - vg) ** 2 / y, fvmax.expand_as(vg)),
        ),
    )
    gain = -force_scale(gp) * _fl_curve(lg, gp[:, 4], gp[:, 5]) * fv
    l0b = (lr[:, 1] - lr[:, 0]) / (bp[:, 1] - bp[:, 0])
    lb = bp[:, 0] + (length - lr[:, 0]) / l0b
    half = 0.5 * (1.0 + bp[:, 5])
    fp = torch.where(
        lb <= 1.0,
        torch.zeros_like(lb),
        torch.where(
            lb <= half,
            0.5 * ((lb - 1.0) / (half - 1.0)) ** 2,
            0.5 + (lb - half) / (half - 1.0),
        ),
    )
    bias = -force_scale(bp) * bp[:, 7] * fp
    return gain, bias


def coupling_derivatives(qpos: Any, t: dict[str, Any]) -> Any:
    """``(N, C)`` derivative of every dependent coordinate w.r.t. its independent one."""
    dx = qpos[:, t["coupling_ind_qadr"]] - t["coupling_x0"]
    c = t["coupling_coef"]
    return c[:, 1] + dx * (2.0 * c[:, 2] + dx * (3.0 * c[:, 3] + dx * 4.0 * c[:, 4]))


def muscle_matrix(
    ten_j: Any, qpos: Any, gain: Any, bias: Any, t: dict[str, Any]
) -> tuple[Any, Any]:
    """Muscle generalised forces on the static-optimisation rows.

    Args:
        ten_j: ``(N, nJten)`` sparse tendon Jacobian values (model layout).
        qpos: ``(N, nq)`` joint positions (for the coupling derivatives).
        gain: ``(N, M)`` force per unit activation (negative).
        bias: ``(N, M)`` passive force (negative).
        t: Torch tables.

    Returns:
        ``A (N, R, M)`` and ``b (N, R)`` with ``A a + b`` the generalised force the
        muscles put on the rows' hinges at activation ``a`` (coupled coordinates
        included through the coupling velocity map).
    """
    import torch

    n = qpos.shape[0]
    m, k = t["slot_adr"].shape
    r = t["muscle_rows"].shape[0]
    vals = ten_j[:, t["slot_adr"].reshape(-1)].reshape(n, m, k)
    deriv = coupling_derivatives(qpos, t)
    fac_all = torch.cat(
        [torch.ones_like(deriv[:, :1]), deriv, torch.zeros_like(deriv[:, :1])], 1
    )
    vals = vals * fac_all[:, t["slot_fac"].reshape(-1)].reshape(n, m, k)
    a_src = (vals * gain[:, :, None]).reshape(n, -1)
    b_src = (vals * bias[:, :, None]).reshape(n, -1)
    muscle_idx = torch.arange(m, device=qpos.device)[:, None].expand(m, k)
    flat = (t["slot_row"] * m + muscle_idx).reshape(-1)
    a_flat = torch.zeros(n, (r + 1) * m, dtype=vals.dtype, device=vals.device)
    a_flat.index_add_(1, flat, a_src)
    b_rows = torch.zeros(n, r + 1, dtype=vals.dtype, device=vals.device)
    b_rows.index_add_(1, t["slot_row"].reshape(-1), b_src)
    return a_flat.reshape(n, r + 1, m)[:, :r], b_rows[:, :r]


def static_optimisation(a_n: Any, r_n: Any, c: Any, rho: float, iters: int) -> Any:
    """Box-constrained least-squares muscle sharing, batched.

    Minimises ``|a - c|^2 + rho |A_n a - r_n|^2`` over ``0 <= a <= 1``: the closed-form
    solution on the free set (``a = c + A^T (A A^T + I / rho)^-1 (r - A c)``), with muscles
    that leave the box fixed at the bound they crossed, ``iters`` times.

    Args:
        a_n: ``(N, R, M)`` normalised muscle matrix.
        r_n: ``(N, R)`` normalised torque demand (passive forces removed).
        c: ``(N, M)`` regularisation target in ``[0, 1]`` (co-contraction).
        rho: Residual weight.
        iters: Active-set passes.

    Returns:
        ``(N, M)`` activations in ``[0, 1]``.
    """
    import torch

    n, r, m = a_n.shape
    free = torch.ones(n, m, dtype=torch.bool, device=a_n.device)
    fixed = torch.zeros_like(c)
    eye = torch.eye(r, dtype=a_n.dtype, device=a_n.device) / rho
    a = c
    for _ in range(iters):
        a_f = a_n * free[:, None, :]
        base = torch.where(free, c, fixed)
        rhs = r_n - torch.bmm(a_n, base[..., None])[..., 0]
        h = torch.bmm(a_f, a_f.transpose(1, 2)) + eye
        chol, _ = torch.linalg.cholesky_ex(h)
        y = torch.cholesky_solve(rhs[..., None], chol)
        a = base + torch.bmm(a_f.transpose(1, 2), y)[..., 0]
        low = free & (a < 0.0)
        high = free & (a > 1.0)
        fixed = torch.where(
            low,
            torch.zeros_like(fixed),
            torch.where(high, torch.ones_like(fixed), fixed),
        )
        free = free & ~(low | high)
    return a.clamp(0.0, 1.0)


def excitation_for(
    a_star: Any, act: Any, tau_act: Any, tau_deact: Any, h: float, n_steps: int
) -> Any:
    """Excitation that moves the activation to ``a_star`` in ``n_steps`` Euler steps of ``h``.

    MuJoCo's muscle activation: ``act' = (u - act) / tau`` with
    ``tau = tau_act (0.5 + 1.5 act)`` when ``u > act`` and ``tau_deact / (0.5 + 1.5 act)``
    otherwise; ``tau`` is frozen at the current activation over the horizon. The result is
    clipped to ``[0, 1]``, so deactivation is never faster than the model allows.
    """
    import torch

    act = act.clamp(0.0, 1.0)
    up = a_star > act
    tau = torch.where(up, tau_act * (0.5 + 1.5 * act), tau_deact / (0.5 + 1.5 * act))
    e = (1.0 - h / tau).clamp(min=0.0) ** n_steps
    u = (a_star - e * act) / (1.0 - e).clamp(min=1e-6)
    return u.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# mjlab action term
# ---------------------------------------------------------------------------


class CoordinatedMuscleActionCfg:
    """Action term config: joint-space PD targets through the muscle coordination layer.

    The action is ``[a_q (H hinges), a_c (5 co-contraction groups), (residual, M)]``.

    Args:
        entity_name: Scene entity key.
        mj_model: The compiled subject model the scene is built from.
        coord: Layer settings.
        clips: The task's clip bank (the reference the PD targets are offsets from).
        ctrl_dt: Control step (s).
        feedforward: Per clip, the tables of :func:`~.feedforward.bank_feedforward`
            (``tau``: static feed-forward hinge torques, ``inertia``: joint-space inertia,
            each ``(T, H)``). Required by the ``"inertia"`` gain mode.
        frames_qpos: Poses over which hinge torque capacity is measured (default: frames
            of *clips*).
    """

    def __init__(
        self,
        *,
        entity_name: str,
        mj_model: mujoco.MjModel,
        coord: CoordinationCfg,
        clips: tuple[Any, ...],
        ctrl_dt: float,
        frames_qpos: np.ndarray | None = None,
        feedforward: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.entity_name = entity_name
        self.feedforward = feedforward
        if feedforward is not None and len(feedforward) != len(clips):
            raise ValueError("one feed-forward table per clip")
        if coord.gain_mode not in ("bounded", "inertia", "teacher"):
            raise ValueError(f"unknown gain_mode {coord.gain_mode!r}")
        if coord.gain_mode != "teacher" and feedforward is None:
            raise ValueError(
                f"the {coord.gain_mode!r} gain mode needs the feed-forward tables (inertia)"
            )
        self.coord = coord
        self.clips = clips
        self.ctrl_dt = ctrl_dt
        if frames_qpos is None:
            frames_qpos = sample_frames(clips, 48)
        self.tables = build_tables(mj_model, coord, frames_qpos)
        self.tendon_names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_TENDON, int(i))
            for i in self.tables.muscle_tendons
        ]
        self.sim_dt = float(mj_model.opt.timestep)

    def build(self, env: Any) -> CoordinatedMuscleAction:
        return CoordinatedMuscleAction(self, env)


class CoordinatedMuscleAction:
    """mjlab action term of :class:`CoordinatedMuscleActionCfg` (see the module docstring)."""

    def __init__(self, cfg: CoordinatedMuscleActionCfg, env: Any) -> None:
        import torch

        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        tb = cfg.tables
        self._t = torch_tables(tb, self.device)
        self._entity = env.scene[cfg.entity_name]
        tendon_ids, resolved = self._entity.find_tendons(
            cfg.tendon_names, preserve_order=True
        )
        if list(resolved) != list(cfg.tendon_names):
            raise ValueError("muscle tendon order does not resolve")
        self._tendon_ids = torch.tensor(
            tendon_ids, device=self.device, dtype=torch.long
        )
        neck_ids, resolved = self._entity.find_joints(
            list(sb.NECK_JOINTS), preserve_order=True
        )
        if list(resolved) != list(sb.NECK_JOINTS):
            raise ValueError("neck joints do not resolve in order")
        self._neck_joint_ids = torch.tensor(
            neck_ids, device=self.device, dtype=torch.long
        )
        self.n_hinges = len(tb.hinge_names)
        self.n_groups = len(tb.group_names)
        self.n_muscles = len(tb.muscle_actuators)
        self.n_res = self.n_muscles if cfg.coord.residual_scale > 0 else 0
        self.n_tau = self.n_hinges if cfg.coord.tau_res_scale > 0 else 0
        dim = self.n_hinges + self.n_groups + self.n_tau + self.n_res
        self._raw = torch.zeros(self.num_envs, dim, device=self.device)
        self._q_tgt = torch.zeros(self.num_envs, self.n_hinges, device=self.device)
        self._qd_tgt = torch.zeros_like(self._q_tgt)
        self._cocon = torch.zeros(self.num_envs, self.n_muscles, device=self.device)
        self._res = torch.zeros(self.num_envs, self.n_muscles, device=self.device)
        self._tau_res = torch.zeros(self.num_envs, self.n_hinges, device=self.device)
        tau_unit = np.zeros(self.n_hinges, np.float32)
        tau_unit[tb.muscle_rows] = tb.capacity
        tau_unit[tb.neck_rows] = tb.neck_gear
        self._tau_unit = cfg.coord.tau_res_scale * torch.as_tensor(
            tau_unit, device=self.device
        )
        self._u = torch.zeros(self.num_envs, self.n_muscles, device=self.device)
        self._neck = torch.zeros(self.num_envs, 3, device=self.device)
        self._tau_ff = torch.zeros(self.num_envs, self.n_hinges, device=self.device)
        self._inertia = torch.ones(self.num_envs, self.n_hinges, device=self.device)
        self._qacc_ref = torch.zeros(self.num_envs, self.n_hinges, device=self.device)
        self._tables: dict[str, Any] = {}
        if cfg.feedforward is not None:
            dt = float(cfg.ctrl_dt)
            per_clip = {
                "tau": [cfg.coord.ff_scale * f["tau"] for f in cfg.feedforward],
                "inertia": [f["inertia"] for f in cfg.feedforward],
                "qacc": [
                    np.gradient(np.asarray(c.qvel)[:, tb.hinge_dof], dt, axis=0)
                    if c.qvel is not None and len(c.qvel) > 1
                    else np.zeros((len(c.qpos), self.n_hinges))
                    for c in cfg.clips
                ],
            }
            t_max = max(len(c.qpos) for c in cfg.clips)
            for key, arrays in per_clip.items():
                pad = np.zeros((len(arrays), t_max, self.n_hinges), np.float32)
                for i, a in enumerate(arrays):
                    pad[i, : len(a)] = a
                    pad[i, len(a) :] = a[-1]
                self._tables[key] = torch.as_tensor(pad, device=self.device)
        self._substep = 0
        self._decimation = int(env.cfg.decimation)
        self._n_horizon = max(1, int(round(cfg.coord.horizon_s / cfg.sim_dt)))
        self._diag_sum = torch.zeros(6, device=self.device)
        self._diag_n = 0

    @property
    def action_dim(self) -> int:
        return self._raw.shape[1]

    @property
    def raw_action(self) -> Any:
        return self._raw

    def _reference(self) -> tuple[Any, Any]:
        """Reference hinge positions / velocities ``target_lead_steps`` steps ahead; also
        loads that frame's feed-forward torque, inertia and reference acceleration."""
        import torch

        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            _resolve_mimic_mjlab_ids,
        )

        cache = _resolve_mimic_mjlab_ids(
            self._env,
            self.cfg.entity_name,
            "fullbody",
            self.cfg.clips,
            self.cfg.ctrl_dt,
        )
        src = cache["clip_source"]
        step = self._env.episode_length_buf + int(self.cfg.coord.target_lead_steps)
        frames = src.frame_indices(step)
        ids = torch.arange(self.num_envs, device=self.device)
        q = src.ref_qpos_at_frames(frames, ids)
        qd = src.ref_qvel_at_frames(frames, ids)
        t = self._t
        if self._tables:
            clip_idx = getattr(src, "_clip_indices", None)
            if clip_idx is None:
                clip_idx = torch.zeros_like(frames)
            self._tau_ff[:] = self._tables["tau"][clip_idx, frames]
            self._inertia[:] = self._tables["inertia"][clip_idx, frames]
            self._qacc_ref[:] = self._tables["qacc"][clip_idx, frames]
        return q[:, t["hinge_qadr"]].float(), qd[:, t["hinge_dof"]].float()

    def process_actions(self, actions: Any) -> None:
        import torch

        c = self.cfg.coord
        t = self._t
        self._raw[:] = actions.to(self.device)
        h, g = self.n_hinges, self.n_groups
        q_ref, qd_ref = self._reference()
        self._q_tgt[:] = torch.clamp(
            q_ref + c.dq_scale * self._raw[:, :h], t["hinge_lo"], t["hinge_hi"]
        )
        self._qd_tgt[:] = qd_ref
        level = (c.cocon_base + c.cocon_scale * self._raw[:, h : h + g]).clamp(
            0.0, c.cocon_max
        )
        self._cocon[:] = level[:, t["muscle_group"]]
        if self.n_tau:
            self._tau_res[:] = self._tau_unit * self._raw[:, h + g : h + g + self.n_tau]
        if self.n_res:
            self._res[:] = c.residual_scale * self._raw[:, h + g + self.n_tau :]
        self._substep = 0

    def _solve(self, record: bool) -> None:
        import torch

        c = self.cfg.coord
        t = self._t
        data = self._env.sim.data
        qpos, qvel = data.qpos, data.qvel
        q = qpos[:, t["hinge_qadr"]]
        qd = qvel[:, t["hinge_dof"]]
        if c.gain_mode == "inertia":
            kp = self._inertia * c.omega**2
            kd = self._inertia * (2.0 * c.zeta * c.omega)
        elif c.gain_mode == "bounded":
            kp = torch.minimum(t["kp"], self._inertia * c.omega_max**2)
            kd = 2.0 * c.zeta * torch.sqrt(kp * self._inertia)
        else:
            kp, kd = t["kp"], t["kd"]
        tau = (
            self._tau_ff
            + self._tau_res
            + kp * (self._q_tgt - q)
            + kd * (self._qd_tgt - qd)
        )
        if c.accel_ff and self._tables:
            tau = tau + self._inertia * self._qacc_ref
        # Neck motors.
        nr = t["neck_rows"]
        self._neck[:] = (tau[:, nr] / t["neck_gear"]).clamp(-1.0, 1.0)
        # Muscles.
        ma = t["muscle_actuators"]
        length = data.actuator_length[:, ma]
        velocity = data.actuator_velocity[:, ma]
        gain, bias = muscle_gain_bias(length, velocity, t)
        a_mat, b = muscle_matrix(data.ten_J, qpos, gain, bias, t)
        rows = t["muscle_rows"]
        scale = (t["capacity"] * t["active_rows"]).clamp(min=1e-6)[None, :, None]
        a_n = a_mat / scale * t["active_rows"][None, :, None]
        r_n = ((tau[:, rows] - b) / scale[..., 0]) * t["active_rows"]
        a_star = static_optimisation(a_n, r_n, self._cocon, c.rho, c.active_set_iters)
        act = data.act[:, ma]
        u = excitation_for(
            a_star, act, t["tau_act"], t["tau_deact"], self.cfg.sim_dt, self._n_horizon
        )
        if self.n_res:
            u = (u + self._res).clamp(0.0, 1.0)
        self._u[:] = u
        if record:
            with torch.no_grad():
                resid = (torch.bmm(a_n, a_star[..., None])[..., 0] - r_n).abs().mean()
                demand = r_n.abs().mean()
                self._diag_sum += torch.stack(
                    [
                        resid,
                        demand,
                        a_star.mean(),
                        (a_star > 0.99).float().mean(),
                        (self._q_tgt - q).pow(2).mean().sqrt(),
                        self._cocon.mean(),
                    ]
                )
                self._diag_n += 1

    def apply_actions(self) -> None:
        if self._substep == 0 or self.cfg.coord.every_substep:
            self._solve(record=self._substep == 0)
        self._substep += 1
        self._entity.set_tendon_effort_target(self._u, tendon_ids=self._tendon_ids)
        self._entity.set_joint_effort_target(self._neck, joint_ids=self._neck_joint_ids)

    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0

    def diagnostics(self) -> dict[str, float]:
        """Means since the last call: torque residual and demand (normalised), mean and
        saturated activation, PD error RMS (rad), co-contraction level."""
        if self._diag_n == 0:
            return {}
        v = (self._diag_sum / self._diag_n).tolist()
        self._diag_sum.zero_()
        self._diag_n = 0
        keys = ("torque_residual", "torque_demand", "act_star_mean", "act_saturated",
                "pd_err_rms", "cocon_mean")  # fmt: skip
        return dict(zip(keys, v))
