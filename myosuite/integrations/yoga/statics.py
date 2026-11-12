# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Can the muscles hold it? Static feasibility of yoga holds for MyoFullBody.

At a hold exemplar ``q`` (zero velocity) a linear program minimises the peak
muscle activation ``s`` subject to static equilibrium of the independent
coordinates (couplings' constraint forces drop out after projecting with the
velocity map ``T``)::

    T^T [ M^T (g * a) + sum_k J_k^T D_k lam_k + S tau + L ell ] = T^T [ b - p - M^T h ]
    0 <= a <= s,  lam >= 0,  |tau| <= 1,  ell >= 0

``M`` is the muscle moment matrix, ``g`` / ``h`` the active (at full activation)
and passive muscle forces at this pose (force-length, zero velocity), ``J_k`` the
Jacobians of floor contacts within ``margin``, ``D_k`` friction-pyramid edges,
``S`` the neck motors (gear), ``L`` unilateral joint-limit reactions for joints at
a range limit, ``b`` gravity and ``p`` passive joint springs. ``s* > 1`` means the
muscles cannot hold the pose statically; an infeasible LP means no static
equilibrium exists on these contacts at all.

CLI::

    python -m myosuite.integrations.yoga.statics --table subject_table.json \
        --clips-dir <retargeted clips> --holds <release_holds.yaml> --out statics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.yoga import subject_body as sb
from myosuite.integrations.yoga.kinematics import ReducedKinematics


def load_holds(release_holds: Path) -> list[dict]:
    """Annotated holds of a release manifest: stem, name, group, 60 Hz exemplar frame, ground zones."""
    import yaml

    doc = yaml.safe_load(Path(release_holds).read_text())
    out = []
    for clip in doc["clips"]:
        for h in clip.get("holds", []):
            out.append({
                "stem": clip["stem"], "group": clip.get("group", ""), "name": h["name"],
                "frame": int(h["frame_hold"]), "ground": list(h.get("pairs_ground", [])),
            })  # fmt: skip
    return out


def _moment_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    dense = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        dense,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    return dense


def hold_statics(
    model: mujoco.MjModel,
    rk: ReducedKinematics,
    q: np.ndarray,
    mu: float = 0.8,
    margin: float = 0.01,
    limit_tol: float = np.radians(0.5),
    use_limits: bool | Collection[str] = True,
    return_solution: bool = False,
    passive_scale: float = 1.0,
    braces: bool = True,
    brace_margin: float = 0.01,
    qvel: np.ndarray | None = None,
    qacc: np.ndarray | None = None,
) -> dict:
    """Minimum peak activation that holds ``q`` statically (see module docstring).

    ``use_limits``: ``True`` lets every hinge at a range limit take a free, rigid limit
    reaction; a collection of joint names restricts that to those joints (e.g. the ones
    with a stiff end-range, ``SubjectTable.limit_solref``); ``False`` allows none.

    ``qvel`` / ``qacc`` make it the dynamic version: the load gains ``M(q) qacc`` and the
    velocity terms, and the muscles act at their shortening velocity (force-velocity).
    """
    from scipy.optimize import linprog

    d = mujoco.MjData(model)
    d.qpos[:] = q
    if qvel is not None:
        d.qvel[:] = qvel
    mujoco.mj_forward(model, d)
    T = rk.velocity_map(q)
    mus = np.nonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
    gain = np.array([
        mujoco.mju_muscleGain(d.actuator_length[i], d.actuator_velocity[i], model.actuator_lengthrange[i], model.actuator_acc0[i], model.actuator_gainprm[i, :9])
        for i in mus
    ])  # fmt: skip
    bias = np.array([
        mujoco.mju_muscleBias(d.actuator_length[i], model.actuator_lengthrange[i], model.actuator_acc0[i], model.actuator_biasprm[i, :9])
        for i in mus
    ])  # fmt: skip
    bias = passive_scale * bias
    Mm = _moment_matrix(model, d)[mus]
    cols = [T.T @ (Mm.T * gain)]
    # Floor contacts within the margin, pyramidal friction.
    edges = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]], float)
    n_con = n_brace = 0
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 != 0 and b2 != 0:
            # Brace: equal and opposite forces; the contact frame's normal points 1 -> 2.
            if not braces or c.dist > brace_margin:
                continue
            fr = np.asarray(c.frame).reshape(3, 3)
            D = np.array(
                [fr[0] + sgn * mu * fr[k] for k in (1, 2) for sgn in (1.0, -1.0)]
            )
            j1, j2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, d, j1, None, c.pos, b1)
            mujoco.mj_jac(model, d, j2, None, c.pos, b2)
            cols.append(T.T @ (j2 - j1).T @ D.T)
            n_con += 1
            n_brace += 1
            continue
        g = g2 if model.geom_bodyid[g1] == 0 else g1
        if c.dist > margin:
            continue
        jp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jp, None, c.pos, int(model.geom_bodyid[g]))
        cols.append(T.T @ jp.T @ edges.T)
        n_con += 1
    neck = [
        i
        for i in range(model.nu)
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
    ]
    if neck:
        S = np.zeros((model.nv, len(neck)))
        for k, i in enumerate(neck):
            S[model.jnt_dofadr[model.actuator_trnid[i, 0]], k] = model.actuator_gear[
                i, 0
            ]
        cols.append(T.T @ S)
    n_lim = 0
    if use_limits:
        L = []
        for adr, dof, lo, hi in zip(
            rk.indep_hinge_qadr,
            rk.indep_hinge_dof,
            rk.indep_hinge_lo,
            rk.indep_hinge_hi,
        ):
            if (
                use_limits is not True
                and model.joint(int(model.dof_jntid[dof])).name not in use_limits
            ):
                continue
            for bound, sign in ((lo, 1.0), (hi, -1.0)):
                if abs(q[adr] - bound) < limit_tol:
                    e = np.zeros(model.nv)
                    e[dof] = sign
                    L.append(e)
        if L:
            cols.append(T.T @ np.array(L).T)
            n_lim = len(L)
    A = np.hstack(cols)
    load = d.qfrc_bias - d.qfrc_passive - Mm.T @ bias
    if qacc is not None:
        inertial = np.zeros(model.nv)
        mujoco.mj_mulM(model, d, inertial, np.asarray(qacc, dtype=float))
        load = load + inertial
    rhs = T.T @ load
    n_a, n_x = len(mus), A.shape[1]
    A_eq = np.hstack([A, np.zeros((A.shape[0], 1))])
    A_ub = np.hstack([np.eye(n_a), np.zeros((n_a, n_x - n_a)), -np.ones((n_a, 1))])
    bounds = [(0, None)] * n_a + [(0, None)] * (4 * n_con)
    bounds += [(-1, 1)] * len(neck) + [(0, None)] * n_lim + [(0, None)]
    cost = np.zeros(n_x + 1)
    cost[-1] = 1.0
    res = linprog(
        cost,
        A_ub=A_ub,
        b_ub=np.zeros(n_a),
        A_eq=A_eq,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )
    out = {
        "status": int(res.status),
        "contacts": n_con,
        "braces": n_brace,
        "limits": n_lim,
        "s_star": float(res.x[-1]) if res.status == 0 else None,
    }
    if res.status == 0:
        a = res.x[:n_a]
        top = np.argsort(a)[-5:][::-1]
        out["top_muscles"] = {
            model.actuator(int(mus[i])).name: round(float(a[i]), 3) for i in top
        }
        out["passive_share"] = float(
            np.abs(Mm.T @ bias).sum() / max(np.abs(d.qfrc_bias).sum(), 1e-9)
        )
        if return_solution:
            out["activation"] = dict(zip(mus.tolist(), a.tolist()))
            k0 = n_a + 4 * n_con
            out["neck_ctrl"] = dict(zip(neck, res.x[k0 : k0 + len(neck)].tolist()))
    return out


def normalised_fibre_lengths(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """``(T, n_muscles)`` normalised fibre length ``(l - l_slack) / l_opt`` along a trajectory."""
    mus = np.nonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
    lr, rng = model.actuator_lengthrange[mus], model.actuator_gainprm[mus, :2]
    l_opt = (lr[:, 1] - lr[:, 0]) / (rng[:, 1] - rng[:, 0])
    l_slack = lr[:, 0] - rng[:, 0] * l_opt
    d = mujoco.MjData(model)
    out = np.empty((len(qpos), len(mus)))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_tendon(model, d)
        mujoco.mj_transmission(model, d)
        out[t] = (d.actuator_length[mus] - l_slack) / l_opt
    return out


def fit_passive_compliance(
    model: mujoco.MjModel,
    clips_dir: Path,
    cap: float = 1.25,
    quantile: float = 99.0,
    stride: int = 6,
) -> dict[str, float]:
    """``fpmax`` factors that give her corpus stretches the passive cost of ``cap``.

    For each muscle whose ``quantile`` normalised fibre length over the corpus exceeds
    ``cap``, the factor makes the passive force at that stretch equal the stock passive
    force at ``cap``. Muscles that stay below ``cap`` are untouched.
    """
    mus = np.nonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
    L = np.concatenate(
        [
            normalised_fibre_lengths(model, np.load(p)[::stride])
            for p in sorted(Path(clips_dir).glob("*.qpos60.npy"))
        ]
    )
    peak = np.percentile(L, quantile, axis=0)
    out = {}
    for k, i in enumerate(mus):
        if peak[k] <= cap:
            continue
        prm, lr, acc0 = (
            model.actuator_biasprm[i, :9],
            np.array([0.0, 1.0]),
            float(model.actuator_acc0[i]),
        )
        unit = np.array(
            [0.0, 1.0, 1.0, 1.0, *prm[4:9]]
        )  # l_opt = 1, l_slack = 0: length is normalised
        f_cap = -mujoco.mju_muscleBias(cap, lr, acc0, unit)
        f_peak = -mujoco.mju_muscleBias(float(peak[k]), lr, acc0, unit)
        out[model.actuator(int(i)).name] = float(f_cap / f_peak) if f_peak > 0 else 1.0
    return out


def _job(args: tuple) -> list[dict]:
    table_path, clips_dir, holds, margin = args
    table = sb.SubjectTable.load(Path(table_path))
    model = sb.compile_spec(sb.build_subject_spec(table))
    model.geom_margin[:] = (
        margin  # supports within this of the floor count (the plant LP used 2 cm)
    )
    rk = ReducedKinematics(model)
    out = []
    cache: dict[str, np.ndarray] = {}
    for h in holds:
        p = Path(clips_dir) / f"{h['stem']}.qpos60.npy"
        if not p.exists():
            continue
        Q = cache.setdefault(h["stem"], np.load(p))
        q = Q[min(h["frame"], len(Q) - 1)]
        r = {**h, **hold_statics(model, rk, q, margin=margin)}
        r["s_star_no_limits"] = hold_statics(
            model, rk, q, margin=margin, use_limits=False
        )["s_star"]
        r["s_star_no_passive"] = hold_statics(
            model, rk, q, margin=margin, passive_scale=0.0
        )["s_star"]
        if table.limit_solref:
            r["s_star_stiff_limits"] = hold_statics(
                model, rk, q, margin=margin, use_limits=set(table.limit_solref)
            )["s_star"]
        out.append(r)
    return out


def main() -> None:
    """Run the LP over every annotated hold and print a summary."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--holds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="contact distance (m) counted as support",
    )
    ap.add_argument(
        "--write-compliant",
        type=Path,
        default=None,
        help="fit her flexibility (fit_passive_compliance), write that table here and analyse it",
    )
    args = ap.parse_args()
    if args.write_compliant is not None:
        import dataclasses

        table = sb.SubjectTable.load(args.table)
        model = sb.compile_spec(sb.build_subject_spec(table))
        compliance = fit_passive_compliance(model, args.clips_dir)
        dataclasses.replace(table, passive_scale=compliance).save(args.write_compliant)
        args.table = args.write_compliant
    holds = load_holds(args.holds)
    chunks = [holds[i :: args.workers] for i in range(args.workers)]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.workers) as pool:
        jobs = [(str(args.table), str(args.clips_dir), c, args.margin) for c in chunks]
        res = [r for part in pool.map(_job, jobs) for r in part]
    res.sort(key=lambda r: (r["stem"], r["frame"]))
    args.out.write_text(json.dumps(res, indent=1))
    s = np.array([r["s_star"] if r["s_star"] is not None else np.inf for r in res])
    s_nl = np.array(
        [
            r["s_star_no_limits"] if r["s_star_no_limits"] is not None else np.inf
            for r in res
        ]
    )
    s_np = np.array(
        [
            r["s_star_no_passive"] if r["s_star_no_passive"] is not None else np.inf
            for r in res
        ]
    )
    print(f"holds {len(res)}: no static equilibrium {np.isinf(s).sum()}, muscles saturate (s*>1) {((s > 1) & np.isfinite(s)).sum()}, "
          f"holdable {(s <= 1).sum()} (without joint-limit reactions {(s_nl <= 1).sum()}; "
          f"without passive muscle force {(s_np <= 1).sum()})")  # fmt: skip
    if "s_star_stiff_limits" in res[0]:
        s_st = np.array(
            [
                r["s_star_stiff_limits"]
                if r["s_star_stiff_limits"] is not None
                else np.inf
                for r in res
            ]
        )
        print(
            f"  with limit reactions only on the stiff end-ranges: holdable {(s_st <= 1).sum()}"
        )
    by_group: dict[str, list[float]] = {}
    for r, v in zip(res, s):
        by_group.setdefault(r["group"], []).append(v)
    for g, v in sorted(by_group.items()):
        v = np.array(v)
        print(
            f"  {g:14s} n {len(v):3d}  holdable {(v <= 1).mean():.2f}  median s* {np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else float('nan'):.2f}"
        )


if __name__ == "__main__":
    main()
