# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate a yoga tracker checkpoint on every clip, on mjlab.

Each clip is played from its first frame (``--start first``), from its quietest
mid-clip frame (``--start hold``; the reference keeps moving after it), or as a static
reference frozen at that frame (``--start frozen``: can the body hold the pose?), or from
every multiple of ``--sweep-step-s`` seconds (``--start sweep``: which parts of a clip are
learnt?) until it deviates (any tracked site more than ``max_site_err`` from its target),
the clip ends or ``--horizon-s`` passes. ``--at-s`` replaces the quietest frame by a chosen
time for ``hold`` and ``frozen``, and ``--include`` keeps only the named clips. ``--render``
selects videos by substring, e.g. ``@26s`` for one start of a sweep.
``--action`` picks the controller: the policy's mean action (``mean``), samples
from its distribution as in training (``sampled``), or the mean action with the
muscle sigmoid flattened so that each excitation equals its expectation under the
training noise (``matched``, see :func:`noise_matched_action`). ``--repeats``
plays every clip that many times and averages. Writes per-clip survival and
tracking error, and renders videos of the first replica with the reference drawn
as a ghost. ``--record-state`` also saves, per rendered clip, the replica's
``qpos/qvel/act/ctrl/qacc/actuator_force`` at every control step (``<clip>.state.npz``,
read before the step, so ``ctrl`` is the previous step's) and the env's compiled model
(``model.mjb``), for offline dynamics (:mod:`myosuite.integrations.yoga.dynamics_viz`).

CLI::

    python -m myosuite.integrations.yoga.evaluate --run <run dir> --checkpoint model_N.pt \
        --refs <ref dir> --plant-xml <plant.xml> --out <dir> [--render Crane Handstand]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.core.trajectory_io import MotionClip

# Per-step state saved by ``--record-state`` (besides ``qpos``), read before each step.
STATE_KEYS = ("qvel", "act", "ctrl", "qacc", "actuator_force")


def _force_starts(
    env: Any, entity: str, clip_ids: list[int], offsets: list[int]
) -> None:
    """Put env ``i`` on clip ``clip_ids[i]`` at frame ``offsets[i]`` (state and targets)."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _resolve_mimic_mjlab_ids,
    )

    cache = _resolve_mimic_mjlab_ids(env, entity, "fullbody")
    src = cache["clip_source"]
    dev = env.device
    ids = torch.arange(env.num_envs, device=dev)
    if hasattr(src, "_clip_indices"):  # a clip bank; one clip uses a single-clip source
        src._clip_indices[:] = torch.as_tensor(clip_ids, device=dev)
    elif any(clip_ids):
        raise ValueError("a single-clip source can only play clip 0")
    src._start_offsets[:] = torch.as_tensor(offsets, device=dev)
    src._last_step[:] = 0
    q = src.ref_qpos_at_frames(src._start_offsets, ids).float()
    v = src.ref_qvel_at_frames(src._start_offsets, ids).float()
    root = torch.cat(
        [
            q[:, :3] + env.scene.env_origins,
            q[:, 3:7],
            v[:, :3],
            quat_apply(q[:, 3:7], v[:, 3:6]),
        ],
        -1,
    )
    ent = env.scene[entity]
    ent.write_root_state_to_sim(root, env_ids=ids)
    ent.write_joint_state_to_sim(q[:, 7:], v[:, 6:], env_ids=ids)
    env.sim.forward()
    cache["synced_inputs"] = None


def run_task_kwargs(cfg_json: dict) -> dict:
    """Task settings of a run from its ``config.json`` (action space and coordination)."""
    out: dict = {}
    args = cfg_json.get("args", {})
    if "effort_weight" in args:
        out["effort_weight"] = float(args["effort_weight"])
    coord = cfg_json.get("coord")
    if coord:
        from myosuite.integrations.yoga.coordination import CoordinationCfg

        out.update(action_space="joint", coord=CoordinationCfg(**coord))
    return out


def hold_start(clip: MotionClip, lo: float = 0.2, hi: float = 0.8) -> int:
    """The most quasi-static frame (lowest joint-speed norm) in the middle of a clip."""
    a, b = int(lo * len(clip.qvel)), int(hi * len(clip.qvel))
    return a + int(np.argmin(np.linalg.norm(clip.qvel[a:b], axis=1)))


MEAN_SITE_TERMINATION_M = 1.0
ROOT_TERMINATION_M = 0.3


def _pelvis_frame_error(
    sites: Any, targets: Any, root_pos: Any, root_quat: Any, ref_root: Any
) -> np.ndarray:
    """Per-site error with each body's root pose removed, ``(N, n_sites)`` in metres.

    ``ref_root`` is the reference root ``(N, 7)`` (position, wxyz quaternion).
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    n, s = sites.shape[:2]

    def local(points: Any, pos: Any, quat: Any) -> Any:
        q = quat[:, None, :].expand(n, s, 4).reshape(-1, 4)
        return quat_apply_inverse(q, (points - pos[:, None, :]).reshape(-1, 3))

    cur = local(sites, root_pos, root_quat)
    tgt = local(targets, ref_root[:, :3], ref_root[:, 3:7])
    return (cur - tgt).norm(dim=-1).reshape(n, s).cpu().numpy()


def noise_matched_action(
    policy: Any, obs: Any, n_muscles: int, gain: float = 5.0
) -> Any:
    """Deterministic action whose muscle excitations match the noisy policy's mean.

    Muscles map an action ``a`` to ``sigmoid(gain * (a - 0.5))``. Under the policy's
    Gaussian noise ``a ~ N(mu, s^2)`` the expected excitation is, by the probit
    approximation, ``sigmoid(gain * (mu - 0.5) / sqrt(1 + pi * gain^2 * s^2 / 8))``:
    the noise flattens the sigmoid and lifts silent muscles. Returns ``mu`` with its
    first ``n_muscles`` entries rescaled about 0.5 by that factor; the remaining
    (linear) actions keep their mean.
    """
    import math

    import torch

    mu = policy(obs)
    dist = policy.distribution
    if dist.std_type == "scalar":
        std = dist.std_param.clamp(dist.std_range[0], dist.std_range[1])
    else:
        std = dist.log_std_param.clamp(*dist.log_std_range).exp()
    scale = torch.rsqrt(1.0 + math.pi * (gain * std[:n_muscles]) ** 2 / 8.0)
    out = mu.clone()
    out[:, :n_muscles] = 0.5 + (mu[:, :n_muscles] - 0.5) * scale
    return out


def frozen_reference(ref: Any, at_s: float, seconds: float) -> Any:
    """The plant reference held still at ``at_s`` for ``seconds`` (for the ghost overlay)."""
    from myosuite.integrations.yoga.reference_io import YogaReference

    j = min(int(round(at_s * ref.fps)), ref.n_frames - 1)
    n = int(seconds * ref.fps) + 2
    return YogaReference(
        ref.name,
        ref.fps,
        np.repeat(ref.body_pos[j : j + 1], n, axis=0),
        np.repeat(ref.body_quat[j : j + 1], n, axis=0),
    )


def rollout(
    run: Path,
    checkpoint: Path,
    table: Path,
    clips_dir: Path,
    max_site_err: float = 0.5,
    start: str = "first",
    horizon_s: float | None = None,
    action: str = "mean",
    repeats: int = 1,
    best_trace: bool = False,
    include: list[str] | None = None,
    at_s: float | None = None,
    sweep_step_s: float = 2.0,
    record_state: bool = False,
) -> dict:
    """Play every clip from frame 0 (``start="first"``), from its quietest mid-clip
    frame (``"hold"``), frozen at that frame (``"frozen"``) or from every multiple of
    ``sweep_step_s`` seconds (``"sweep"``, one entry ``<clip>@<t>s`` per start) until it
    deviates, the clip ends or ``horizon_s`` passes (default 10 s for frozen clips).
    ``at_s`` (seconds) replaces the quietest frame; ``include`` keeps only clips whose
    name contains one of its substrings.

    Env ``k`` plays clip ``k % n_clips``; per-clip metrics average the ``repeats``
    replicas. ``action`` is ``"mean"``, ``"sampled"`` or ``"matched"`` (module
    docstring). Returns per clip the qpos trace of its first replica (of its longest-
    surviving one with ``best_trace``), start frames and per-clip
    metrics. ``record_state`` adds ``states`` (per clip, the same replica's
    :data:`STATE_KEYS` traces) and ``mj_model`` (the env's compiled model)."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _clip_ref,
        _resolve_mimic_mjlab_ids,
        _tracked_site_pos,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        FULLBODY_BODY2SITES_FOR_MIMIC,
    )
    from myosuite.integrations.yoga.task import (
        CTRL_DT,
        ENTITY,
        TASK_ID,
        clip_names,
        frozen_clip,
        register_yoga_task,
        start_sweep,
    )

    cfg_json = json.loads((run / "config.json").read_text())
    task_kw = run_task_kwargs(cfg_json)
    if action == "matched" and task_kw.get("action_space") == "joint":
        raise ValueError(
            "the noise-matched controller applies to the muscle action space only"
        )
    hold_frames: list[int] = []

    def hold_frame(clip: MotionClip) -> int:
        if at_s is None:
            return hold_start(clip)
        return min(int(round(at_s * float(clip.frequency_hz))), len(clip.qpos) - 1)

    def prepare(bank: tuple[MotionClip, ...]) -> tuple[MotionClip, ...]:
        if start == "sweep":
            return start_sweep(bank, sweep_step_s)
        hold_frames.extend(hold_frame(c) for c in bank)
        seconds = horizon_s if horizon_s is not None else 10.0
        return tuple(frozen_clip(c, f, seconds) for c, f in zip(bank, hold_frames))

    clips = register_yoga_task(
        table,
        clips_dir,
        num_envs=1,
        max_episode_steps=100000,
        max_site_err=max_site_err,
        include=include,
        clip_fn=prepare if start in ("frozen", "sweep") else None,
        **task_kw,
    )
    names = clip_names(clips)
    n = len(clips)
    n_envs = n * repeats
    clip_of = np.arange(n_envs) % n
    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = n_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
    wrapped = RslRlVecEnvWrapper(env)
    n_muscles = (
        env.action_manager.get_term("muscles").action_dim
        if "muscles" in env.action_manager.active_terms
        else 0
    )
    runner = MjlabOnPolicyRunner(
        env=wrapped, train_cfg=cfg_json["runner"], log_dir=None, device="cuda"
    )
    with torch.inference_mode():
        runner.load(str(checkpoint), map_location="cuda")
    policy = runner.get_inference_policy(device="cuda")
    wrapped.reset()
    offsets = [hold_frame(c) if start == "hold" else 0 for c in clips]
    reported_starts = hold_frames if start == "frozen" else offsets
    _force_starts(env, ENTITY, clip_of.tolist(), [offsets[c] for c in clip_of])
    obs = wrapped.get_observations()
    lengths = np.array([len(c.qpos) for c in clips]) - np.array(offsets)
    if horizon_s is not None:
        lengths = np.minimum(lengths, int(horizon_s / CTRL_DT))
    lengths = lengths[clip_of]
    alive = np.ones(n_envs, bool)
    survived = np.zeros(n_envs, int)
    err_sum = np.zeros(n_envs)
    traces = [[] for _ in range(n_envs)]
    states = [{k: [] for k in STATE_KEYS} for _ in range(n_envs)]
    # Termination margins at the last surviving step: worst site / max_site_err, mean site
    # error / MEAN_SITE_TERMINATION_M, root error / ROOT_TERMINATION_M. The largest names
    # the rule that ended the episode one control step later.
    margins = np.zeros((n_envs, 3))
    worst_site = np.zeros(n_envs, int)
    # The worst site's error with the root pose removed (pelvis frame), over its world error:
    # near 0 when the body moves as a whole (tips or slides), near 1 when the posture breaks.
    posture_share = np.zeros(n_envs)
    failed = np.zeros(n_envs, bool)
    site_names = list(FULLBODY_BODY2SITES_FOR_MIMIC.values())
    ent = env.scene[ENTITY]
    with torch.inference_mode():
        for t in range(int(lengths.max()) - 1):
            qpos = ent.data.data.qpos.cpu().numpy()
            extra = (
                {k: getattr(ent.data.data, k).cpu().numpy() for k in STATE_KEYS}
                if record_state
                else {}
            )
            for i in np.nonzero(alive)[0]:
                traces[i].append(qpos[i].copy())
                for k, v in extra.items():
                    states[i][k].append(v[i].copy())
            if action == "matched":
                act = noise_matched_action(policy, obs, n_muscles)
            else:
                act = policy(obs, stochastic_output=action == "sampled")
            obs, _, dones, _ = wrapped.step(act)
            cache = _resolve_mimic_mjlab_ids(env, ENTITY, "fullbody")
            per_site = (
                (_tracked_site_pos(env, ENTITY, cache) - cache["target_torch"])
                .norm(dim=-1)
                .cpu()
                .numpy()
            )
            ref_root = _clip_ref(env, cache, "ref_qpos")[:, :7].float()
            root_err = (
                (ent.data.root_link_pos_w - ref_root[:, :3]).norm(dim=-1).cpu().numpy()
            )
            local = _pelvis_frame_error(
                _tracked_site_pos(env, ENTITY, cache),
                cache["target_torch"],
                ent.data.root_link_pos_w,
                ent.data.root_link_quat_w,
                ref_root,
            )
            err = per_site.mean(-1)
            d = dones.cpu().numpy().astype(bool)
            step_ok = alive & ~d
            failed |= alive & d & (t + 2 < lengths)
            err_sum[step_ok] += err[step_ok]
            survived[step_ok] += 1
            margins[step_ok] = np.stack(
                [
                    per_site.max(-1) / max_site_err,
                    err / MEAN_SITE_TERMINATION_M,
                    root_err / ROOT_TERMINATION_M,
                ],
                -1,
            )[step_ok]
            worst_site[step_ok] = per_site.argmax(-1)[step_ok]
            w = per_site.argmax(-1)
            rows = np.arange(n_envs)
            share = local[rows, w] / np.maximum(per_site[rows, w], 1e-6)
            posture_share[step_ok] = share[step_ok]
            alive &= ~d
            alive &= t + 2 < lengths
            if not alive.any():
                break
    completed = np.minimum(1.0, (survived + 2) / lengths)
    site_err = err_sum / np.maximum(survived, 1)
    metrics = {}
    for i, name in enumerate(names):
        k = clip_of == i
        metrics[name] = {
            "clip_s": float(lengths[i] * CTRL_DT), "survived_s": float(survived[k].mean() * CTRL_DT),
            "completed": float(completed[k].mean()), "mean_site_err_m": float(site_err[k].mean()),
        }  # fmt: skip
        if repeats > 1:
            metrics[name]["survived_s_replicas"] = (survived[k] * CTRL_DT).tolist()
        causes: dict[str, int] = {}
        for e in np.nonzero(k & failed)[0]:
            rule = int(np.argmax(margins[e]))
            if rule == 0:
                how = "posture" if posture_share[e] >= 0.5 else "whole_body"
                key = f"site:{site_names[worst_site[e]]}:{how}"
            else:
                key = ("mean", "root")[rule - 1]
            causes[key] = causes.get(key, 0) + 1
        metrics[name]["end_causes"] = causes
    mj_model = env.sim.mj_model if record_state else None
    env.close()
    chosen = [
        int(np.flatnonzero(clip_of == i)[np.argmax(survived[clip_of == i])])
        if best_trace
        else i
        for i in range(n)
    ]
    out = {
        "names": names,
        "traces": [np.asarray(traces[e]) for e in chosen],
        "trace_replicas": [int(e // n) for e in chosen],
        "offsets": reported_starts,
        "metrics": metrics,
    }
    if record_state:
        out["states"] = [
            {k: np.asarray(v) for k, v in states[e].items()} for e in chosen
        ]
        out["mj_model"] = mj_model
    return out


def main() -> None:
    """Evaluate one checkpoint and render selected clips."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--plant-xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--render",
        nargs="*",
        default=None,
        help="clip substrings to render (none: no videos)",
    )
    ap.add_argument(
        "--start", choices=("first", "hold", "frozen", "sweep"), default="first"
    )
    ap.add_argument("--horizon-s", type=float, default=None)
    ap.add_argument("--action", choices=("mean", "sampled", "matched"), default="mean")
    ap.add_argument(
        "--repeats", type=int, default=1, help="replicas per clip (averaged)"
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="clip substrings to evaluate (default: all)",
    )
    ap.add_argument(
        "--at-s",
        type=float,
        default=None,
        help="hold/frozen start time (default: quietest frame)",
    )
    ap.add_argument("--sweep-step-s", type=float, default=2.0)
    ap.add_argument(
        "--render-best",
        action="store_true",
        help="render each clip's longest-surviving replica instead of the first",
    )
    ap.add_argument(
        "--record-state",
        action="store_true",
        help="save each rendered replica's state per step and the env's model",
    )
    args = ap.parse_args()
    cfg = json.loads((args.run / "config.json").read_text())["args"]
    table, clips_dir = Path(cfg["table"]), Path(cfg["clips_dir"])
    res = rollout(
        args.run, args.checkpoint, table, clips_dir, float(cfg.get("max_site_err", 0.5)),
        start=args.start, horizon_s=args.horizon_s, action=args.action, repeats=args.repeats,
        best_trace=args.render_best, include=args.include, at_s=args.at_s,
        sweep_step_s=args.sweep_step_s, record_state=args.record_state,
    )  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    m = res["metrics"]
    comp = np.array([v["completed"] for v in m.values()])
    summary = {"checkpoint": str(args.checkpoint), "start": args.start, "horizon_s": args.horizon_s,
               "at_s": args.at_s,
               "action": args.action, "repeats": args.repeats,
               "mean_completed": float(comp.mean()), "clips_fully_completed": int((comp >= 0.99).sum()),
               "start_frames": dict(zip(res["names"], res["offsets"])),
               "rendered_replica": dict(zip(res["names"], res["trace_replicas"])), "per_clip": m}  # fmt: skip
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=1))
    for name, v in sorted(m.items(), key=lambda kv: -kv[1]["completed"]):
        print(
            f"{name[:62]:62s} survived {v['survived_s']:5.1f}/{v['clip_s']:5.1f} s  err {100 * v['mean_site_err_m']:4.1f} cm"
        )
    print(
        f"mean completed {comp.mean():.3f}; fully completed {(comp >= 0.99).sum()}/{len(comp)}"
    )
    if args.render:
        from myosuite.integrations.yoga import subject_body as sb
        from myosuite.integrations.yoga.reference_io import (
            load_reference,
            plant_colliders,
        )
        from myosuite.integrations.yoga.render import render_retarget

        model = sb.compile_spec(sb.build_subject_spec(sb.SubjectTable.load(table)))
        cols = plant_colliders(args.plant_xml)
        note = (
            f" (best of {args.repeats} replicas)"
            if args.render_best and args.repeats > 1
            else ""
        )
        if args.record_state:
            import mujoco

            mujoco.mj_saveModel(res["mj_model"], str(args.out / "model.mjb"), None)
        for i, (name, tr, off) in enumerate(
            zip(res["names"], res["traces"], res["offsets"])
        ):
            if not any(s in name for s in args.render) or len(tr) < 10:
                continue
            base, at = name.rsplit("@", 1) if args.start == "sweep" else (name, None)
            ref, start_s = load_reference(args.refs / f"{base}.npz"), off / 100.0
            if at is not None:  # sweep entry <clip>@<t>s: the reference starts at t
                start_s = float(at[:-1])
            if args.start == "frozen":
                ref, start_s = frozen_reference(ref, start_s, len(tr) / 100.0), 0.0
            np.save(args.out / f"{name}.qpos.npy", tr)  # the rendered trace, 100 Hz
            if args.record_state:
                np.savez(args.out / f"{name}.state.npz", qpos=tr, start_s=start_s,
                         hz=100.0, **res["states"][i])  # fmt: skip
            render_retarget(model, tr, 100.0, ref, cols, args.out / f"{name}.mp4",
                            title=f"policy {args.checkpoint.stem}: {name}{note}", start_s=start_s)  # fmt: skip


if __name__ == "__main__":
    main()
