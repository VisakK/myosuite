# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reduced (independent) coordinates of a model with joint-coupling equalities.

MyoFullBody drives 37 of its 88 DOFs independently; the others (vertebrae, shoulder
girdle, knee translations, patella) follow polynomial ``<equality joint>``
couplings. MuJoCo's convention (verified by :meth:`ReducedKinematics.coupling_residual`)::

    q[joint1] - q0[joint1] = sum_k c_k (q[joint2] - q0[joint2])**k      (k = 0..4)

with ``joint2`` absent for a locked joint. Treating the couplings as exact
functions keeps inverse kinematics and resampling on the constraint manifold.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class _Coupling:
    dep_qadr: int
    dep_dof: int
    ind_qadr: int  # -1 for a locked joint
    ind_dof: int
    coef: np.ndarray  # (5,)
    y0: float
    x0: float


class ReducedKinematics:
    """Independent-coordinate view of a MuJoCo model.

    Args:
        model: Compiled model. Only active joint equalities count as couplings.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.couplings: list[_Coupling] = []
        dependent = set()
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[e]:
                continue
            j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            dependent.add(j1)
            self.couplings.append(
                _Coupling(
                    dep_qadr=int(model.jnt_qposadr[j1]), dep_dof=int(model.jnt_dofadr[j1]),
                    ind_qadr=int(model.jnt_qposadr[j2]) if j2 >= 0 else -1,
                    ind_dof=int(model.jnt_dofadr[j2]) if j2 >= 0 else -1,
                    coef=model.eq_data[e, :5].copy(), y0=float(model.qpos0[model.jnt_qposadr[j1]]),
                    x0=float(model.qpos0[model.jnt_qposadr[j2]]) if j2 >= 0 else 0.0,
                )
            )  # fmt: skip
        for c in self.couplings:
            if c.ind_qadr >= 0 and any(
                c.ind_qadr == d.dep_qadr for d in self.couplings
            ):
                raise ValueError("chained joint couplings are not supported")
        self.dependent_joints = sorted(dependent)
        indep_dof: list[int] = []
        hinge_qadr, hinge_dof, lo, hi = [], [], [], []
        for j in range(model.njnt):
            if j in dependent:
                continue
            t = int(model.jnt_type[j])
            dof = int(model.jnt_dofadr[j])
            if t == mujoco.mjtJoint.mjJNT_FREE:
                indep_dof.extend(range(dof, dof + 6))
            elif t == mujoco.mjtJoint.mjJNT_BALL:
                indep_dof.extend(range(dof, dof + 3))
            else:
                indep_dof.append(dof)
                hinge_qadr.append(int(model.jnt_qposadr[j]))
                hinge_dof.append(dof)
                r = model.jnt_range[j] if model.jnt_limited[j] else (-np.inf, np.inf)
                lo.append(float(r[0]))
                hi.append(float(r[1]))
        self.indep_dof = np.array(indep_dof)
        self.indep_hinge_qadr = np.array(hinge_qadr)
        self.indep_hinge_dof = np.array(hinge_dof)
        self.indep_hinge_lo = np.array(lo)
        self.indep_hinge_hi = np.array(hi)
        self.n_indep = len(indep_dof)

    def apply_couplings(self, qpos: np.ndarray) -> np.ndarray:
        """Copy of *qpos* with every dependent coordinate recomputed."""
        q = np.array(qpos, dtype=float, copy=True)
        for c in self.couplings:
            dx = q[c.ind_qadr] - c.x0 if c.ind_qadr >= 0 else 0.0
            q[c.dep_qadr] = c.y0 + float(np.polyval(c.coef[::-1], dx))
        return q

    def clamp(self, qpos: np.ndarray) -> np.ndarray:
        """Clamp independent hinges/slides into their ranges, then re-apply couplings."""
        q = np.array(qpos, dtype=float, copy=True)
        q[self.indep_hinge_qadr] = np.clip(
            q[self.indep_hinge_qadr], self.indep_hinge_lo, self.indep_hinge_hi
        )
        return self.apply_couplings(q)

    def velocity_map(self, qpos: np.ndarray) -> np.ndarray:
        """``(nv, n_indep)`` map from independent velocities to the full ``qvel``."""
        T = np.zeros((self.model.nv, self.n_indep))
        col = {d: i for i, d in enumerate(self.indep_dof)}
        for d, i in col.items():
            T[d, i] = 1.0
        for c in self.couplings:
            if c.ind_qadr < 0:
                continue
            dx = qpos[c.ind_qadr] - c.x0
            deriv = sum(k * c.coef[k] * dx ** (k - 1) for k in range(1, 5))
            T[c.dep_dof, col[c.ind_dof]] = deriv
        return T

    def coupling_residual(self, qpos: np.ndarray) -> float:
        """Max |equality violation| MuJoCo reports at *qpos* (0 when consistent)."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)
        rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        return float(np.abs(data.efc_pos[rows]).max()) if rows.any() else 0.0
