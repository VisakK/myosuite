# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Body-relative observation of the yoga tracker, after ProtoMotions' privileged mimic inputs.

The three inputs of ProtoMotions' ``mlp_2x`` experiment, for the muscle body:

* ``max_coords_obs`` -> :class:`BodyStateObs`: root height, the other bodies relative to the
  root, every body's rotation (x and z axes), linear and angular velocity;
* ``mimic_target_poses`` -> :class:`TargetPoseObs`: per future reference frame and body, the
  position relative to the present root and to the present body, the rotation, the rotation
  relative to the present one, and the velocity differences;
* ``previous_actions`` -> mjlab's ``last_action``.

Everything is turned about the vertical into the root's heading, so the observation does not
depend on where or facing which way the motion is performed. Two additions for this body:
:class:`PreviewObs` (body positions further ahead) and :class:`MuscleStateObs`.

The heading comes from the root's *lateral* axis (forward = up x right). The forward axis
used by ProtoMotions points straight down in a forward fold, where its heading is undefined
and flips; the lateral axis stays near horizontal through folds and inversions.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# Bodies that carry the plant's 24 colliders; the root (pelvis) first.
OBS_BODIES: tuple[str, ...] = (
    "pelvis", "lumbar3", "lumbar1", "torso", "head",
    "clavicle_r", "humerus_r", "ulna_r", "lunate_r",
    "clavicle_l", "humerus_l", "ulna_l", "lunate_l",
    "femur_r", "tibia_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "calcn_l", "toes_l",
)  # fmt: skip
LATERAL_AXIS = 2  # the pelvis' z axis points to the subject's right
# Columns of a reference body row: position, x axis, z axis, linear and angular velocity.
REF_DIM = 15
TARGET_DIM = 24  # per body and future frame
MUSCLE_VEL_CLIP = 3.0  # normalised fibre velocity (the curve is flat beyond 1)
MUSCLE_FORCE_CLIP = 3.0  # in units of peak isometric force


class BodyReference:
    """Reference states of the observed bodies for a clip bank.

    A plain class, so that config serialisers do not expand its arrays.

    Attributes:
        body_names: Observed bodies, root first.
        bank: ``(sum of clip lengths, B, 15)`` rows of :data:`REF_DIM` per body, all clips
            concatenated in time.
        starts: ``(n_clips,)`` row of each clip's first frame.
        clips: The clip bank the rows were computed from.
        ctrl_dt: Control step (s).
    """

    def __init__(
        self,
        body_names: tuple[str, ...],
        bank: np.ndarray,
        starts: np.ndarray,
        clips: tuple[Any, ...],
        ctrl_dt: float,
    ) -> None:
        self.body_names = body_names
        self.bank = bank
        self.starts = starts
        self.clips = clips
        self.ctrl_dt = ctrl_dt

    def __repr__(self) -> str:
        return f"BodyReference({len(self.body_names)} bodies, {len(self.bank)} frames, {len(self.clips)} clips)"

    def __deepcopy__(self, memo: dict) -> BodyReference:
        return self  # read-only; mjlab deep-copies term configs


def body_kinematics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None,
    body_ids: np.ndarray,
) -> np.ndarray:
    """Body states along a trajectory by forward kinematics.

    Args:
        model: The subject model.
        qpos: ``(T, nq)`` positions.
        qvel: ``(T, nv)`` velocities (``None``: at rest).
        body_ids: ``(B,)`` body ids.

    Returns:
        ``(T, B, 15)`` float32: world position of the body frame, its x and z axes, and the
        linear (at the frame origin) and angular velocity, world-aligned.
    """
    data = mujoco.MjData(model)
    root = model.body_rootid[body_ids]
    out = np.zeros((len(qpos), len(body_ids), REF_DIM), np.float32)
    for f in range(len(qpos)):
        data.qpos[:] = qpos[f]
        data.qvel[:] = 0.0 if qvel is None else qvel[f]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mat = data.xmat[body_ids].reshape(-1, 3, 3)
        pos = data.xpos[body_ids]
        ang, lin_c = data.cvel[body_ids, :3], data.cvel[body_ids, 3:]
        out[f, :, 0:3] = pos
        out[f, :, 3:6] = mat[:, :, 0]
        out[f, :, 6:9] = mat[:, :, 2]
        # cvel is taken at the subtree centre of mass: move it to the body frame origin.
        out[f, :, 9:12] = lin_c - np.cross(ang, data.subtree_com[root] - pos)
        out[f, :, 12:15] = ang
    return out


def build_body_reference(
    model: mujoco.MjModel,
    clips: tuple[Any, ...],
    ctrl_dt: float,
    body_names: tuple[str, ...] = OBS_BODIES,
) -> BodyReference:
    """The reference body states of every clip of a bank.

    Args:
        model: The subject model the clips were solved on.
        clips: Clip bank (``qpos`` required, ``qvel`` optional).
        ctrl_dt: Control step (s); the clips play one frame per step.
        body_names: Observed bodies, root first.

    Returns:
        The bank of reference body states.
    """
    ids = np.array([model.body(n).id for n in body_names])
    parts = [body_kinematics(model, c.qpos, c.qvel, ids) for c in clips]
    lengths = np.array([len(p) for p in parts])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return BodyReference(
        tuple(body_names), np.concatenate(parts), starts, tuple(clips), ctrl_dt
    )


def heading_cos_sin(lateral: Any) -> tuple[Any, Any]:
    """Cosine and sine of the heading whose forward direction is ``up x lateral``.

    Args:
        lateral: ``(..., 3)`` world direction of the root's lateral (right) axis.
    """
    import torch

    fx, fy = -lateral[..., 1], lateral[..., 0]
    norm = torch.sqrt(fx * fx + fy * fy).clamp_min(1e-6)
    return fx / norm, fy / norm


def to_heading(v: Any, cos: Any, sin: Any) -> Any:
    """World vectors ``(N, ..., 3)`` turned about the vertical into the heading frame."""
    import torch

    while cos.dim() < v.dim() - 1:
        cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)
    x, y, z = v.unbind(dim=-1)
    return torch.stack([cos * x + sin * y, cos * y - sin * x, z], dim=-1)


def body_state_obs(pos: Any, rot: Any, lin: Any, ang: Any) -> Any:
    """ProtoMotions' max-coords body observation.

    Args:
        pos: ``(N, B, 3)`` body positions, root first.
        rot: ``(N, B, 3, 3)`` body rotation matrices.
        lin: ``(N, B, 3)`` linear velocities.
        ang: ``(N, B, 3)`` angular velocities.

    Returns:
        ``(N, 1 + 3 (B - 1) + 12 B)``: root height, positions relative to the root, x and z
        axes, linear and angular velocities, in the root's heading.
    """
    import torch

    n = pos.shape[0]
    cos, sin = heading_cos_sin(rot[:, 0, :, LATERAL_AXIS])
    axes = torch.cat(
        [to_heading(rot[..., 0], cos, sin), to_heading(rot[..., 2], cos, sin)], dim=-1
    )
    parts = [
        pos[:, 0, 2:3],
        to_heading(pos[:, 1:] - pos[:, :1], cos, sin).reshape(n, -1),
        axes.reshape(n, -1),
        to_heading(lin, cos, sin).reshape(n, -1),
        to_heading(ang, cos, sin).reshape(n, -1),
    ]
    return torch.cat(parts, dim=-1)


def target_pose_obs(pos: Any, rot: Any, lin: Any, ang: Any, ref: Any) -> Any:
    """ProtoMotions' mimic target poses (with relative parts and velocities).

    Args:
        pos: ``(N, B, 3)`` present body positions, root first.
        rot: ``(N, B, 3, 3)`` present body rotation matrices.
        lin: ``(N, B, 3)`` present linear velocities.
        ang: ``(N, B, 3)`` present angular velocities.
        ref: ``(N, S, B, 15)`` reference body rows of ``S`` future frames.

    Returns:
        ``(N, S * 24 * B)``, per frame: target positions relative to the present root and
        to the present bodies, target axes in the heading frame and in the present body
        frames, linear and angular velocity differences.
    """
    import torch

    n, s = ref.shape[:2]
    cos, sin = heading_cos_sin(rot[:, 0, :, LATERAL_AXIS])
    world_to_body = rot.transpose(-1, -2)[:, None]

    def head(v: Any) -> Any:
        return to_heading(v, cos, sin)

    def local(v: Any) -> Any:
        return (world_to_body @ v.unsqueeze(-1)).squeeze(-1)

    r_pos, r_x, r_z = ref[..., 0:3], ref[..., 3:6], ref[..., 6:9]
    parts = [
        head(r_pos - pos[:, None, :1]),
        head(r_pos - pos[:, None]),
        torch.cat([head(r_x), head(r_z)], dim=-1),
        torch.cat([local(r_x), local(r_z)], dim=-1),
        head(ref[..., 9:12] - lin[:, None]),
        head(ref[..., 12:15] - ang[:, None]),
    ]
    return torch.cat([p.reshape(n, s, -1) for p in parts], dim=-1).reshape(n, -1)


def preview_obs(pos: Any, rot: Any, ref_pos: Any) -> Any:
    """Future reference body positions relative to the present root, in its heading.

    Args:
        pos: ``(N, B, 3)`` present body positions, root first.
        rot: ``(N, B, 3, 3)`` present body rotation matrices.
        ref_pos: ``(N, P, B, 3)`` reference positions of ``P`` future frames.

    Returns:
        ``(N, P * 3 * B)``.
    """
    cos, sin = heading_cos_sin(rot[:, 0, :, LATERAL_AXIS])
    return to_heading(ref_pos - pos[:, None, :1], cos, sin).reshape(pos.shape[0], -1)


class _BodyTerm:
    """Shared set-up of the body observation terms (an mjlab class-based term)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        import torch

        self._env = env
        ref: BodyReference = cfg.params["ref"]
        self._entity = env.scene[cfg.params["entity_name"]]
        ids, names = self._entity.find_bodies(ref.body_names, preserve_order=True)
        if tuple(names) != tuple(ref.body_names):
            raise ValueError("the observed bodies do not resolve in order")
        dev = env.device
        self._ids = torch.tensor(ids, device=dev, dtype=torch.long)
        self._bank = torch.as_tensor(ref.bank, device=dev)
        self._starts = torch.as_tensor(ref.starts, device=dev, dtype=torch.long)
        self._steps = torch.tensor(
            cfg.params.get("steps", ()), device=dev, dtype=torch.long
        )

    def reset(self, env_ids: Any = None) -> None:
        del env_ids  # stateless

    def _present(self) -> tuple[Any, Any, Any, Any]:
        """Present body positions (clip frame), rotation matrices and velocities."""
        from mjlab.utils.lab_api.math import matrix_from_quat

        data = self._entity.data
        pose = data.body_link_pose_w[:, self._ids]
        vel = data.body_link_vel_w[:, self._ids]
        pos = pose[..., :3] - self._env.scene.env_origins[:, None, :]
        return pos, matrix_from_quat(pose[..., 3:7]), vel[..., :3], vel[..., 3:]

    def _future(self, ref: BodyReference, entity_name: str) -> Any:
        """Reference body rows ``(N, S, B, 15)`` at the term's future steps."""
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            _resolve_mimic_mjlab_ids,
        )

        cache = _resolve_mimic_mjlab_ids(
            self._env, entity_name, "fullbody", ref.clips, ref.ctrl_dt
        )
        src = cache["clip_source"]
        step = self._env.episode_length_buf
        last = src.clip_lengths(step) - 1
        frames = src.frame_indices(step)[:, None] + self._steps[None, :]
        rows = frames.minimum(last[:, None])  # the last frame is held past the clip end
        clip = getattr(src, "_clip_indices", None)
        if clip is not None:
            rows = rows + self._starts[clip][:, None]
        return self._bank[rows]


class BodyStateObs(_BodyTerm):
    """Observation term: :func:`body_state_obs` of the present bodies."""

    def __call__(self, env: Any, ref: BodyReference, entity_name: str) -> Any:
        return body_state_obs(*self._present())


class TargetPoseObs(_BodyTerm):
    """Observation term: :func:`target_pose_obs` for the frames ``steps`` ahead."""

    def __call__(
        self, env: Any, ref: BodyReference, entity_name: str, steps: tuple[int, ...]
    ) -> Any:
        return target_pose_obs(*self._present(), self._future(ref, entity_name))


class PreviewObs(_BodyTerm):
    """Observation term: :func:`preview_obs` for the frames ``steps`` ahead."""

    def __call__(
        self, env: Any, ref: BodyReference, entity_name: str, steps: tuple[int, ...]
    ) -> Any:
        pos, rot, _, _ = self._present()
        return preview_obs(pos, rot, self._future(ref, entity_name)[..., 0:3])


def muscle_scales(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Per-muscle constants that normalise length, velocity and force.

    Returns:
        ``ids`` (actuator ids), ``act`` (activation addresses), ``lo`` and ``l0`` (tendon
        length at the lower end of the operating range, and the optimal fibre length),
        ``range0`` (normalised length there), ``vmax`` (m/s) and ``f0`` (N).
    """
    ids = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    gp, lr = model.actuator_gainprm[ids], model.actuator_lengthrange[ids]
    l0 = (lr[:, 1] - lr[:, 0]) / (gp[:, 1] - gp[:, 0])
    f0 = np.where(gp[:, 2] > 0, gp[:, 2], gp[:, 3] / np.maximum(model.actuator_acc0[ids], 1e-10))  # fmt: skip
    return {
        "ids": ids, "act": model.actuator_actadr[ids], "lo": lr[:, 0], "l0": l0,
        "range0": gp[:, 0], "vmax": l0 * gp[:, 6], "f0": f0,
    }  # fmt: skip


class MuscleStateObs:
    """Observation term: every muscle's activation, and with ``full`` its normalised fibre
    length, velocity (clipped) and force (in peak isometric force, clipped)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        import torch

        scales = muscle_scales(env.sim.mj_model)
        self._c = {
            k: torch.as_tensor(
                v, device=env.device, dtype=torch.long if k in ("ids", "act") else torch.float32
            )
            for k, v in scales.items()
        }  # fmt: skip

    def reset(self, env_ids: Any = None) -> None:
        del env_ids  # stateless

    def __call__(self, env: Any, entity_name: str, full: bool = True) -> Any:
        import torch

        c = self._c
        # No entity.data equivalent for the muscle state (accepted raw reads).
        data = env.scene[entity_name].data.data
        act = data.act[:, c["act"]]
        if not full:
            return act.clone()
        length = c["range0"] + (data.actuator_length[:, c["ids"]] - c["lo"]) / c["l0"]
        velocity = data.actuator_velocity[:, c["ids"]] / c["vmax"]
        force = -data.actuator_force[:, c["ids"]] / c["f0"]
        return torch.cat(
            [
                act,
                length,
                velocity.clamp(-MUSCLE_VEL_CLIP, MUSCLE_VEL_CLIP),
                force.clamp(-MUSCLE_FORCE_CLIP, MUSCLE_FORCE_CLIP),
            ],
            dim=-1,
        )


def body_observation_groups(
    entity_name: str,
    ref: BodyReference,
    phase_fn: Any,
    target_steps: tuple[int, ...] = (1,),
    preview_steps: tuple[int, ...] = (5, 10, 20, 50, 100),
    muscle_state: str = "full",
) -> dict[str, Any]:
    """Observation groups in the ProtoMotions style.

    Args:
        entity_name: Scene entity key.
        ref: Reference body states of the task's clip bank.
        phase_fn: Observation function of the clip phase (critic only).
        target_steps: Future frames (control steps ahead) of the full target block.
        preview_steps: Future frames of the position preview (empty: none).
        muscle_state: ``"full"`` (activation, length, velocity, force), ``"act"`` or
            ``"none"``.

    Returns:
        ``actor``: body state, targets, preview, muscle state, previous action.
        ``critic``: what the critic reads besides the actor group (the clip phase).
    """
    from mjlab.envs.mdp.observations import last_action
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    if muscle_state not in ("full", "act", "none"):
        raise ValueError(
            f"muscle_state must be full, act or none, got {muscle_state!r}"
        )
    base = {"ref": ref, "entity_name": entity_name}
    terms: dict[str, Any] = {
        "body_state": ObservationTermCfg(func=BodyStateObs, params=dict(base)),
        "target_poses": ObservationTermCfg(
            func=TargetPoseObs, params={**base, "steps": tuple(target_steps)}
        ),
    }
    if preview_steps:
        terms["preview"] = ObservationTermCfg(
            func=PreviewObs, params={**base, "steps": tuple(preview_steps)}
        )
    if muscle_state != "none":
        terms["muscle_state"] = ObservationTermCfg(
            func=MuscleStateObs,
            params={"entity_name": entity_name, "full": muscle_state == "full"},
        )
    terms["previous_action"] = ObservationTermCfg(func=last_action)
    return {
        "actor": ObservationGroupCfg(terms=terms),
        "critic": ObservationGroupCfg(
            terms={"clip_phase": ObservationTermCfg(func=phase_fn)}
        ),
    }
