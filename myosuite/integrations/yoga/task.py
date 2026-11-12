# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab (MuJoCo-Warp) task: muscle-driven tracking of the yoga clip bank.

Built on the MuscleMimic full-body mimic task (DeepMimic composite reward,
reference-state initialisation over a clip bank, lookahead observations), with:

* the subject-scaled MyoFullBody (:mod:`.subject_body`) instead of the stock body;
* the three neck torque motors as a second action term (MyoFullBody has no neck);
* a muscle-effort penalty (mean squared activation);
* the PhysX teacher's per-body deviation rule: any tracked site more than
  ``max_site_err`` from its target ends the episode.

The clips are local (MOYO/AMASS-licensed), so the task is registered by
:func:`register_yoga_task` at runtime rather than at import.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.trajectory_io import MotionClip, check_clip_rate, load_motion_clip
from myosuite.integrations.yoga import subject_body as sb

TASK_ID = "myoMimicYoga-v0"
ENTITY = "yoga_subject_robot"
SIM_DT = 0.002
CTRL_DT = 0.01


def load_yoga_clips(
    clips_dir: Path,
    model: mujoco.MjModel,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[MotionClip, ...]:
    """Retargeted clips of *clips_dir* (``*.npz`` written by :mod:`.retarget`).

    Args:
        clips_dir: Directory of clip files.
        model: The subject model the clips were solved on (shape check).
        include: Keep only clips whose name contains one of these substrings.
        exclude: Drop clips whose name contains one of these substrings.

    Returns:
        The clip bank, sorted by name.
    """
    clips = []
    for p in sorted(Path(clips_dir).glob("*.npz")):
        if include and not any(s in p.stem for s in include):
            continue
        if exclude and any(s in p.stem for s in exclude):
            continue
        clip = load_motion_clip(p, expected_nq=model.nq, expected_nv=model.nv)
        check_clip_rate(clip.frequency_hz, CTRL_DT, p.stem)
        clips.append(clip)
    if not clips:
        raise FileNotFoundError(f"no clips selected in {clips_dir}")
    return tuple(clips)


def _muscle_effort(entity_name: str, n_muscles: int) -> Callable[[Any], Any]:
    """Mean squared muscle activation (muscles own the first activation slots)."""

    def _fn(env: Any) -> Any:
        act = env.scene[entity_name].data.data.act[:, :n_muscles]
        return (act * act).mean(dim=-1)

    return _fn


def yoga_env_cfg(
    table: sb.SubjectTable,
    clips: tuple[MotionClip, ...],
    num_envs: int,
    max_episode_steps: int = 1000,
    effort_weight: float = 1.0,
    max_site_err: float = 0.5,
    mimic_reward_weight: float = 5.0,
    action_space: str = "muscle",
    coord: Any = None,
    feedforward: list[np.ndarray] | None = None,
) -> Any:
    """The yoga tracking task as a :class:`mjlab.envs.ManagerBasedRlEnvCfg`.

    Args:
        table: Subject table.
        clips: Clip bank (each env draws a clip and a start frame at reset).
        num_envs: Parallel environments.
        max_episode_steps: Episode cap in control steps (10 s at 100 Hz).
        effort_weight: Weight of the mean squared activation penalty per step.
        max_site_err: Per-site deviation (m) that ends an episode.
        mimic_reward_weight: Weight of the DeepMimic composite reward per step.
        action_space: ``"muscle"``: one excitation per muscle plus the neck torques (the
            original task). ``"joint"``: PD targets (offsets from the reference) and
            co-contraction levels through the muscle coordination layer
            (:mod:`.coordination`).
        coord: :class:`~.coordination.CoordinationCfg` for ``"joint"`` (default settings
            when ``None``).
        feedforward: For ``"joint"``: per clip, the static feed-forward hinge torques
            (:func:`~.feedforward.bank_feedforward`).

    Returns:
        The environment config.
    """
    if action_space not in ("muscle", "joint"):
        raise ValueError(
            f"action_space must be 'muscle' or 'joint', got {action_space!r}"
        )
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.envs.mdp.actions import JointEffortActionCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _make_mimic_env_cfg,
        _muscle_actuator_names,
        _muscle_tendon_names,
        set_mimic_entity_model,
    )

    def spec_fn() -> mujoco.MjSpec:
        return sb.build_subject_spec(table)

    mj = sb.compile_spec(spec_fn(), SIM_DT)
    set_mimic_entity_model(ENTITY, mj)
    muscles, tendons = _muscle_actuator_names(mj), _muscle_tendon_names(mj)
    cfg = _make_mimic_env_cfg(
        _task_id=TASK_ID, entity_name=ENTITY, variant="fullbody", spec_fn=spec_fn,
        muscle_actuators=muscles, tendon_targets=tendons, sim_dt=SIM_DT, ctrl_dt=CTRL_DT,
        max_episode_steps=max_episode_steps, clip=clips, num_envs=num_envs, mj_model=mj,
        mimic_reward_weight=mimic_reward_weight, max_site_err_threshold=max_site_err,
    )  # fmt: skip
    if table.neck:
        entity = cfg.scene.entities[ENTITY]
        entity.articulation.actuators = (
            *entity.articulation.actuators,
            XmlActuatorCfg(
                target_names_expr=sb.NECK_JOINTS,
                transmission_type=TransmissionType.JOINT,
            ),
        )
        cfg.actions["neck"] = JointEffortActionCfg(
            entity_name=ENTITY, actuator_names=sb.NECK_JOINTS, scale=1.0
        )
    if action_space == "joint":
        from myosuite.integrations.yoga.coordination import (
            CoordinatedMuscleActionCfg,
            CoordinationCfg,
        )

        if not table.neck:
            raise ValueError(
                "the joint action space drives the neck motors: table.neck"
            )
        cfg.actions = {
            "joint_pd": CoordinatedMuscleActionCfg(
                entity_name=ENTITY,
                mj_model=mj,
                coord=coord or CoordinationCfg(),
                clips=clips,
                ctrl_dt=CTRL_DT,
                feedforward=feedforward,
            )  # fmt: skip
        }
    # mjlab multiplies reward terms by ctrl_dt; divide so the weight is per step.
    cfg.rewards["effort"] = RewardTermCfg(
        func=_muscle_effort(ENTITY, len(muscles)), weight=-effort_weight / CTRL_DT
    )
    return cfg


def yoga_runner_cfg(
    hidden: tuple[int, ...] = (1024, 512, 256),
    init_std: float = 0.6,
    desired_kl: float = 0.3,
    max_std: float | None = None,
    entropy_coef: float = 0.001,
    num_learning_epochs: int = 4,
    num_mini_batches: int = 4,
    **kwargs: Any,
) -> Any:
    """rsl_rl PPO settings for the yoga bank (wider nets than the single-clip default).

    The adaptive learning rate targets a KL that rsl_rl sums over action dimensions. With
    357 actions the usual 0.01 pins the learning rate at its 1e-5 floor from the first
    iteration (measured); ``desired_kl`` is therefore scaled to the action count
    (0.3 ~ 0.0008 per dimension, a ~0.02 mean shift per update at std 0.5).

    ``max_std`` caps the action noise. Through the sigmoid muscle mapping, noise on a
    silent muscle raises its mean activation (co-contraction the mean action lacks),
    so an uncapped std can grow during training.
    """
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    distribution: dict[str, Any] = {
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "scalar",
    }
    if max_std is not None:
        distribution["std_range"] = (1e-6, max_std)
    actor = RslRlModelCfg(
        hidden_dims=hidden, activation="elu", obs_normalization=True, distribution_cfg=distribution
    )  # fmt: skip
    critic = RslRlModelCfg(
        hidden_dims=hidden,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=num_learning_epochs, num_mini_batches=num_mini_batches, learning_rate=3e-4, schedule="adaptive", desired_kl=desired_kl,
        entropy_coef=entropy_coef, clip_param=0.2, value_loss_coef=1.0, gamma=0.99, lam=0.95, max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False, use_clipped_value_loss=True,
    )  # fmt: skip
    return RslRlOnPolicyRunnerCfg(
        actor=actor, critic=critic, algorithm=algorithm, **kwargs
    )


def frozen_clip(clip: MotionClip, frame: int, seconds: float) -> MotionClip:
    """``clip`` held still at ``frame`` for ``seconds``: a static reference (zero velocity)."""
    n = int(round(seconds * float(clip.frequency_hz))) + 2

    def hold(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else np.repeat(a[frame : frame + 1], n, axis=0)

    qvel = (
        None
        if clip.qvel is None
        else np.zeros((n, clip.qvel.shape[1]), clip.qvel.dtype)
    )
    return dataclasses.replace(
        clip,
        qpos=hold(clip.qpos),
        qvel=qvel,
        site_xpos=hold(clip.site_xpos),
        weights=None,
    )


def start_sweep(
    bank: tuple[MotionClip, ...], step_s: float, min_left_s: float = 1.0
) -> tuple[MotionClip, ...]:
    """Every clip of ``bank`` cut to start at each multiple of ``step_s`` seconds.

    The cut starting at ``t`` is the clip from that frame on, named ``<stem>@<t>s``
    (through ``source_path``), so playing it from its first frame plays the original
    from ``t``. Starts with less than ``min_left_s`` of the clip left are skipped.
    """

    def cut(a: np.ndarray | None, frame: int) -> np.ndarray | None:
        return None if a is None else a[frame:]

    out = []
    for i, clip in enumerate(bank):
        hz = float(clip.frequency_hz)
        stem = (
            Path(clip.source_path).stem if clip.source_path is not None else f"clip{i}"
        )
        for t in np.arange(0.0, len(clip.qpos) / hz - min_left_s, step_s):
            f = int(round(t * hz))
            out.append(
                dataclasses.replace(
                    clip, qpos=clip.qpos[f:], qvel=cut(clip.qvel, f), site_xpos=cut(clip.site_xpos, f),
                    weights=cut(clip.weights, f), source_path=Path(f"{stem}@{t:g}s.npz"),
                )
            )  # fmt: skip
    return tuple(out)


def register_yoga_task(
    table_path: Path,
    clips_dir: Path,
    num_envs: int = 2048,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    clip_fn: Callable[[tuple[MotionClip, ...]], tuple[MotionClip, ...]] | None = None,
    **cfg_kwargs: Any,
) -> tuple[MotionClip, ...]:
    """Register :data:`TASK_ID` with mjlab for this subject table and clip bank.

    Args:
        table_path: Subject table JSON.
        clips_dir: Directory of retargeted clips.
        num_envs: Number of parallel environments.
        include: Keep only clips whose name contains one of these substrings.
        exclude: Drop clips whose name contains one of these substrings.
        clip_fn: Optional map applied to the loaded bank before registration
            (e.g. :func:`frozen_clip` for static-hold evaluation).
        **cfg_kwargs: Forwarded to :func:`yoga_env_cfg`.

    Returns:
        The registered clip bank.
    """
    from mjlab.tasks.registry import register_mjlab_task

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _register_or_replace_mjlab_task,
    )

    table = sb.SubjectTable.load(table_path)
    model = sb.compile_spec(sb.build_subject_spec(table), SIM_DT)
    sources = load_yoga_clips(clips_dir, model, include, exclude)
    clips = clip_fn(sources) if clip_fn is not None else sources
    coord = cfg_kwargs.get("coord")
    if (
        cfg_kwargs.get("action_space") == "joint"
        and "feedforward" not in cfg_kwargs
        and (coord is None or coord.ff_scale != 0.0 or coord.gain_mode != "teacher")
    ):
        from myosuite.integrations.yoga.feedforward import (
            bank_feedforward,
            coord_options,
        )

        cfg_kwargs["feedforward"] = bank_feedforward(
            Path(table_path),
            sources,
            clips,
            Path(clips_dir).parent / "feedforward",
            **coord_options(coord),
        )
    _register_or_replace_mjlab_task(
        register_mjlab_task, task_id=TASK_ID,
        env_cfg=yoga_env_cfg(table, clips, num_envs, **cfg_kwargs),
        play_env_cfg=yoga_env_cfg(table, clips, 1, **cfg_kwargs),
        rl_cfg=yoga_runner_cfg(), runner_cls=None,
    )  # fmt: skip
    return clips


def clip_names(clips: tuple[MotionClip, ...]) -> list[str]:
    """Clip stems in bank order."""
    return [
        Path(c.source_path).stem if c.source_path is not None else f"clip{i}"
        for i, c in enumerate(clips)
    ]


def site_error_stats(errors: np.ndarray) -> dict[str, float]:
    """Summary of a ``(T, n_sites)`` site-error array (metres)."""
    return {
        "mean": float(errors.mean()),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors.max()),
    }
