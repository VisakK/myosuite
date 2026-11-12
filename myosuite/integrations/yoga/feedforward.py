# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Static feed-forward torques for the muscle coordination layer.

A PD law alone gives no gravity compensation: at zero tracking error it asks the muscles
for no torque, so a pose sags until the error builds up, and in a closed chain (hands and
feet on the floor) the error then saturates the weak joints (hip rotation can give ~25 N m
in the downward dog). The static LP of :mod:`.statics` already finds, for every pose with
a static equilibrium on its floor contacts, muscle activations ``a`` (and neck controls)
that hold it. Its torques on the independent hinges,

    tau_ff = T^T R^T (gain(l) * a + bias(l))       (zero velocity)

are the feed-forward of the layer: ``tau_d = tau_ff + PD``. With them the reference is
(nearly) an equilibrium at zero action, and the PD only corrects deviations. Frames
without a static equilibrium (mid-kick-up) are interpolated in time from their
neighbours. Statically infeasible frames (``s* > 1``) keep the LP's torques, which the
muscles cannot fully deliver; the static optimisation saturates there.

Two options make the tables honest about the plant (both off by default, as round 2 ran):

* ``limits="stiff"``: the LP takes free joint-limit reactions only on joints whose table
  gives them a stiff end-range (``SubjectTable.limit_solref``), not on every hinge at a
  limit; MuJoCo's default limits are soft and yield under load.
* ``planted_inertia``: a hinge whose distal segment is on the floor drives the rest of the
  body about its axis, not its own light segment; the inertia stored for the gain cap is
  then that of the body with the distal contacts held (:func:`planted_hinge_inertia`).

Tables are cached per clip in ``cache_dir`` (keyed by the subject table, the LP margin,
these options and the clip's frames).

CLI (precompute for a set of clips)::

    python -m myosuite.integrations.yoga.feedforward --table subject_table.json \
        --clips-dir <retargeted clips> --include Handstand --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

LP_MARGIN = (
    0.02  # supports within 2 cm of the floor count (the statics CLI and plant LP)
)
_VERSION = b"ff2"  # bump when the cached content changes


LIMIT_POLICIES = ("all", "stiff")


def _cache_key(
    table_path: Path,
    clip_qpos: np.ndarray,
    margin: float,
    limits: str = "all",
    planted_inertia: bool = False,
) -> str:
    h = hashlib.sha1()
    h.update(_VERSION)
    h.update(Path(table_path).read_bytes())
    h.update(np.float64(margin).tobytes())
    if limits != "all" or planted_inertia:
        h.update(f"{limits}:{planted_inertia}".encode())
    h.update(np.ascontiguousarray(clip_qpos, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def coord_options(coord: Any) -> dict[str, Any]:
    """:func:`bank_feedforward` options of a ``CoordinationCfg`` (round-2 defaults for ``None``)."""
    return {
        "limits": getattr(coord, "ff_limits", "all"),
        "planted_inertia": bool(getattr(coord, "planted_inertia", False)),
    }


def planted_hinge_inertia(
    model: Any, data: Any, tmap: np.ndarray, full_m: np.ndarray, hinge_cols: np.ndarray,
    hinge_bodies: np.ndarray, free: np.ndarray, root_cols: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Hinge inertias with each hinge's distal floor contacts held.

    For a hinge whose subtree touches the floor (``data.contact``), its effective inertia
    is computed on the subspace {free root, this hinge} (all other hinges locked) under
    those contacts as rigid point constraints (Gauss's principle): the rest of the body
    rotating about the hinge with the distal segment planted. Hinges without distal
    contacts keep *free*; contacts never lower an inertia.

    Args:
        model: Compiled model (``geom_margin`` sets which contacts count).
        data: Data after ``mj_forward`` at the pose.
        tmap: ``(nv, n_indep)`` coupling velocity map.
        full_m: ``(nv, nv)`` joint-space inertia (``mj_fullM``).
        hinge_cols: Independent-coordinate column of each hinge.
        hinge_bodies: Body of each hinge's joint.
        free: Inertia of each hinge without contacts.
        root_cols: The six root columns.
    """
    import mujoco

    pts = []
    for c in data.contact[: data.ncon]:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 == 0) == (b2 == 0):
            continue
        jp = np.zeros((3, model.nv))
        body = b2 if b1 == 0 else b1
        mujoco.mj_jac(model, data, jp, None, c.pos, body)
        pts.append((body, jp @ tmap))
    out = free.copy()
    if not pts:
        return out
    m_red = tmap.T @ full_m @ tmap
    for k, (col, jb) in enumerate(zip(hinge_cols, hinge_bodies)):
        rows = [J for b, J in pts if _in_subtree(model, b, int(jb))]
        if not rows:
            continue
        cols = np.concatenate([root_cols, [col]])
        J = np.vstack(rows)[:, cols]
        m_inv = np.linalg.inv(m_red[np.ix_(cols, cols)])
        a = J @ m_inv @ J.T
        a += 1e-9 * np.trace(a) * np.eye(len(a))
        m_c = m_inv - m_inv @ J.T @ np.linalg.solve(a, J @ m_inv)
        out[k] = max(out[k], 1.0 / max(float(m_c[-1, -1]), 1e-12))
    return out


def _in_subtree(model: Any, body: int, root: int) -> bool:
    while body not in (0, root):
        body = int(model.body_parentid[body])
    return body == root


def _worker(args: tuple) -> list[tuple[int, int, float, np.ndarray, np.ndarray]]:
    """Solve the static LP for a chunk of frames; return (frame, status, s*, tau, inertia).

    ``inertia`` is the diagonal of the joint-space inertia ``T^T M T`` on the hinges: the
    inertia each hinge drives with the root free and no contacts.
    """
    import mujoco
    import torch

    from myosuite.integrations.yoga import coordination as co
    from myosuite.integrations.yoga import statics
    from myosuite.integrations.yoga import subject_body as sb
    from myosuite.integrations.yoga.kinematics import ReducedKinematics

    table_path, frames, qposes, margin, limits, planted = args
    torch.set_num_threads(1)
    table = sb.SubjectTable.load(Path(table_path))
    use_limits = True if limits == "all" else set(table.limit_solref)
    model = sb.compile_spec(sb.build_subject_spec(table))
    model.geom_margin[:] = margin
    rk = ReducedKinematics(model)
    tb = co.build_tables(model, co.CoordinationCfg())
    t = co.torch_tables(tb, "cpu", torch.float64)
    data = mujoco.MjData(model)
    n_h = len(tb.hinge_names)
    col = {int(d): i for i, d in enumerate(rk.indep_dof)}
    hinge_cols = np.array([col[int(d)] for d in tb.hinge_dof])
    hinge_bodies = model.jnt_bodyid[model.dof_jntid[tb.hinge_dof]]
    root_cols = np.arange(6)
    full_m = np.zeros((model.nv, model.nv))
    out = []
    for f, q in zip(frames, qposes):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, full_m, data.qM)
        tmap = rk.velocity_map(q)
        inertia = np.einsum("ik,ij,jk->k", tmap, full_m, tmap)[hinge_cols]
        if planted:
            inertia = planted_hinge_inertia(
                model, data, tmap, full_m, hinge_cols, hinge_bodies, inertia, root_cols
            )
        r = statics.hold_statics(
            model, rk, q, margin=margin, use_limits=use_limits, return_solution=True
        )
        tau = np.full(n_h, np.nan)
        if r["status"] == 0:
            a = np.zeros(len(tb.muscle_actuators))
            for i, v in r["activation"].items():
                a[int(i)] = v
            length = torch.as_tensor(data.actuator_length[None, tb.muscle_actuators])
            gain, bias = co.muscle_gain_bias(length, torch.zeros_like(length), t)
            a_mat, b = co.muscle_matrix(
                torch.as_tensor(data.ten_J[None]),
                torch.as_tensor(data.qpos[None]),
                gain,
                bias,
                t,
            )
            tau[tb.muscle_rows] = (a_mat[0] @ torch.as_tensor(a) + b[0]).numpy()
            for k, act_id in enumerate(tb.neck_actuators):
                tau[tb.neck_rows[k]] = tb.neck_gear[k] * float(
                    r["neck_ctrl"].get(int(act_id), 0.0)
                )
        out.append(
            (int(f), int(r["status"]), float(r["s_star"] or np.nan), tau, inertia)
        )
    return out


def _fill_infeasible(tau: np.ndarray) -> np.ndarray:
    """Linear interpolation in time over rows that are NaN (no static equilibrium)."""
    out = tau.copy()
    ok = ~np.isnan(out[:, 0])
    if not ok.any():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    for j in range(out.shape[1]):
        out[:, j] = np.interp(idx, idx[ok], out[ok, j])
    return out


def clip_feedforward(
    table_path: Path,
    qpos: np.ndarray,
    cache_dir: Path | None = None,
    margin: float = LP_MARGIN,
    workers: int = 16,
    name: str = "clip",
    limits: str = "all",
    planted_inertia: bool = False,
) -> dict[str, np.ndarray]:
    """Per-frame feed-forward torques of one clip.

    Args:
        table_path: Subject table JSON.
        qpos: ``(T, nq)`` clip frames.
        cache_dir: Where tables are cached (``None``: no cache).
        margin: LP contact margin (m).
        workers: Worker processes for the LPs.
        name: Used in the cache file name.
        limits: ``"all"`` (every hinge at a limit gets a free reaction, as in round 2) or
            ``"stiff"`` (only joints with a stiff end-range in the table).
        planted_inertia: Store :func:`planted_hinge_inertia` instead of the free inertia.

    Returns:
        ``tau`` ``(T, H)`` hinge torques (filled), ``inertia`` ``(T, H)`` joint-space inertia
        of each hinge, ``status`` ``(T,)`` LP status (0 solved), ``s_star`` ``(T,)`` peak
        activation (NaN where unsolved).
    """
    if limits not in LIMIT_POLICIES:
        raise ValueError(f"limits must be one of {LIMIT_POLICIES}, got {limits!r}")
    qpos = np.asarray(qpos, dtype=np.float64)
    path = None
    if cache_dir is not None:
        path = (
            Path(cache_dir)
            / f"{name}.ff.{_cache_key(table_path, qpos, margin, limits, planted_inertia)}.npz"
        )
        if path.exists():
            z = np.load(path)
            return {k: z[k] for k in ("tau", "inertia", "status", "s_star")}
    frames = np.arange(len(qpos))
    chunks = [frames[i::workers] for i in range(workers)]
    jobs = [
        (str(table_path), c, qpos[c], margin, limits, planted_inertia)
        for c in chunks
        if len(c)
    ]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = [r for part in pool.map(_worker, jobs) for r in part]
    res.sort(key=lambda r: r[0])
    status = np.array([r[1] for r in res])
    s_star = np.array([r[2] for r in res])
    tau = _fill_infeasible(np.stack([r[3] for r in res]))
    inertia = np.stack([r[4] for r in res])
    out = {"tau": tau, "inertia": inertia, "status": status, "s_star": s_star}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return out


def bank_feedforward(
    table_path: Path,
    sources: tuple[Any, ...],
    bank: tuple[Any, ...],
    cache_dir: Path | None = None,
    margin: float = LP_MARGIN,
    workers: int = 16,
    limits: str = "all",
    planted_inertia: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Feed-forward tables (``tau``, ``inertia``, each ``(T, H)``) for every clip of
    *bank*, derived from its *sources*.

    *bank* may be *sources* itself or clips cut from them (frozen holds, start sweeps):
    each of its frames is matched to a source frame with the same ``qpos`` bytes. A frame
    found in no source is solved on its own. *limits* and *planted_inertia*: see
    :func:`clip_feedforward`.
    """
    from myosuite.integrations.yoga.task import clip_names

    lookup: dict[bytes, np.ndarray] = {}
    for name, c in zip(clip_names(sources), sources):
        q = np.asarray(c.qpos, dtype=np.float64)
        r = clip_feedforward(
            table_path, q, cache_dir, margin, workers, name, limits, planted_inertia
        )
        for row, t, m in zip(q, r["tau"], r["inertia"]):
            lookup[row.tobytes()] = np.stack([t, m])
    out = []
    for c in bank:
        q = np.asarray(c.qpos, dtype=np.float64)
        rows = [lookup.get(r.tobytes()) for r in q]
        missing = [i for i, r in enumerate(rows) if r is None]
        if missing:
            extra = clip_feedforward(
                table_path,
                q[missing],
                None,
                margin,
                min(workers, len(missing)),
                "clip",
                limits,
                planted_inertia,
            )
            for i, t, m in zip(missing, extra["tau"], extra["inertia"]):
                rows[i] = np.stack([t, m])
        tab = np.stack(rows)
        out.append({"tau": tab[:, 0], "inertia": tab[:, 1]})
    return out


def main() -> None:
    """Precompute and cache the feed-forward tables of a set of clips."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--include", nargs="*", default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="default: <clips-dir>/../feedforward",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--margin", type=float, default=LP_MARGIN)
    ap.add_argument(
        "--limits",
        choices=LIMIT_POLICIES,
        default="all",
        help="joints whose limits give free LP reactions",
    )
    ap.add_argument(
        "--planted-inertia",
        action="store_true",
        help="contact-aware hinge inertia for the gain cap",
    )
    args = ap.parse_args()
    from myosuite.integrations.yoga import subject_body as sb
    from myosuite.integrations.yoga.task import SIM_DT, clip_names, load_yoga_clips

    model = sb.compile_spec(
        sb.build_subject_spec(sb.SubjectTable.load(args.table)), SIM_DT
    )
    clips = load_yoga_clips(args.clips_dir, model, args.include)
    cache = args.cache_dir or args.clips_dir.parent / "feedforward"
    for name, c in zip(clip_names(clips), clips):
        r = clip_feedforward(
            args.table,
            c.qpos,
            cache,
            args.margin,
            args.workers,
            name,
            args.limits,
            args.planted_inertia,
        )
        st = r["status"]
        s = r["s_star"][st == 0]
        print(f"{name}: {len(st)} frames, static equilibrium {np.mean(st == 0):.3f}, "
              f"s*>1 {np.mean(s > 1):.3f}, median s* {np.median(s):.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
