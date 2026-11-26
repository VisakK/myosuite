# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The neck held on the reference by a servo.

MyoFullBody has no neck muscles; the subject body's three neck hinges have torque motors
(:func:`.subject_body._add_neck`). :class:`NeckServoAction` is an mjlab action term without
policy outputs: every physics substep a PD on the three hinges drives the motors toward
the reference pose, as the coordination layer does for the joint action space.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from myosuite.integrations.yoga import subject_body as sb


def neck_servo_gains(
    model: mujoco.MjModel, kp: float, zeta: float
) -> tuple[np.ndarray, np.ndarray]:
    """PD gains of the three neck hinges.

    Args:
        model: Compiled subject model with the neck.
        kp: Stiffness (N m / rad).
        zeta: Damping ratio on the head's inertia about each hinge.

    Returns:
        ``(kp, kd)``, each ``(3,)``; ``kd = 2 zeta sqrt(kp I)``.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    dof = [int(model.joint(j).dofadr[0]) for j in sb.NECK_JOINTS]
    inertia = mass[dof, dof]
    return np.full(3, float(kp)), 2.0 * zeta * np.sqrt(kp * inertia)


class NeckServoActionCfg:
    """Action term config: the neck follows the reference; the policy has no output here.

    Args:
        entity_name: Scene entity key.
        mj_model: The compiled subject model the scene is built from.
        clips: The task's clip bank.
        ctrl_dt: Control step (s).
        kp: Stiffness (N m / rad); the motors saturate at their gear.
        zeta: Damping ratio.
        lead_steps: The target is the reference this many control steps ahead.
    """

    def __init__(
        self,
        *,
        entity_name: str,
        mj_model: mujoco.MjModel,
        clips: tuple[Any, ...],
        ctrl_dt: float,
        kp: float = 100.0,
        zeta: float = 1.0,
        lead_steps: int = 1,
    ) -> None:
        self.entity_name = entity_name
        self.clips = clips
        self.ctrl_dt = ctrl_dt
        self.lead_steps = lead_steps
        self.kp, self.kd = neck_servo_gains(mj_model, kp, zeta)
        self.qadr = [int(mj_model.joint(j).qposadr[0]) for j in sb.NECK_JOINTS]
        self.dof = [int(mj_model.joint(j).dofadr[0]) for j in sb.NECK_JOINTS]
        self.gear = [float(mj_model.actuator(a).gear[0]) for a in sb.NECK_ACTUATORS]

    def build(self, env: Any) -> NeckServoAction:
        return NeckServoAction(self, env)


class NeckServoAction:
    """mjlab action term of :class:`NeckServoActionCfg`."""

    def __init__(self, cfg: NeckServoActionCfg, env: Any) -> None:
        import torch

        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self._entity = env.scene[cfg.entity_name]
        ids, resolved = self._entity.find_joints(
            list(sb.NECK_JOINTS), preserve_order=True
        )
        if list(resolved) != list(sb.NECK_JOINTS):
            raise ValueError("neck joints do not resolve in order")
        self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        self._qadr = torch.tensor(cfg.qadr, device=self.device, dtype=torch.long)
        self._dof = torch.tensor(cfg.dof, device=self.device, dtype=torch.long)
        self._kp = torch.tensor(cfg.kp, device=self.device, dtype=torch.float32)
        self._kd = torch.tensor(cfg.kd, device=self.device, dtype=torch.float32)
        self._gear = torch.tensor(cfg.gear, device=self.device, dtype=torch.float32)
        self._q_tgt = torch.zeros(self.num_envs, 3, device=self.device)
        self._qd_tgt = torch.zeros_like(self._q_tgt)
        self._ctrl = torch.zeros_like(self._q_tgt)
        self._raw = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> Any:
        return self._raw

    def process_actions(self, actions: Any) -> None:
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
        frames = src.frame_indices(self._env.episode_length_buf + self.cfg.lead_steps)
        ids = torch.arange(self.num_envs, device=self.device)
        self._q_tgt[:] = src.ref_qpos_at_frames(frames, ids)[:, self._qadr]
        self._qd_tgt[:] = src.ref_qvel_at_frames(frames, ids)[:, self._dof]

    def apply_actions(self) -> None:
        data = self._entity.data
        q = data.joint_pos[:, self._joint_ids]
        qd = data.joint_vel[:, self._joint_ids]
        tau = self._kp * (self._q_tgt - q) + self._kd * (self._qd_tgt - qd)
        self._ctrl[:] = (tau / self._gear).clamp(-1.0, 1.0)
        self._entity.set_joint_effort_target(self._ctrl, joint_ids=self._joint_ids)

    def reset(self, env_ids: Any = None) -> None:
        return None

    def tracking_error(self) -> Any:
        """Neck angle error against the present target, ``(N, 3)`` rad."""
        return self._entity.data.joint_pos[:, self._joint_ids] - self._q_tgt
