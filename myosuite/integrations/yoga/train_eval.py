# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""In-training evaluation of a yoga tracker: frozen holds and start sweeps, mean action.

Runs in the training process on its own (small) mjlab environment, so the scores land in
the training logger (TensorBoard / W&B) at the iteration they belong to. It reproduces the
headline tests of :mod:`.evaluate` (README §4):

* ``frozen``: the reference held still at chosen times for ``horizon_s`` ("can it hold
  the pose?"), ``frozen_replicas`` replicas each;
* ``sweep``: ``horizon_s`` from every ``sweep_step_s`` of the clip ("which parts are
  learnt?"), split into starts inside the named hold windows and all other starts.

Every rollout uses the policy's mean action and ends at the task's own termination rules.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    """What the in-training evaluation plays.

    Attributes:
        frozen_at_s: Times (s) whose frames are frozen for the hold test.
        frozen_replicas: Replicas per frozen hold.
        sweep_step_s: Spacing of the sweep starts (s).
        sweep_replicas: Replicas per sweep start.
        hold_windows_s: ``(start, end)`` windows (s) whose sweep starts count as "hold".
        horizon_s: Length of every rollout (s).
    """

    frozen_at_s: tuple[float, ...] = (19.88,)
    frozen_replicas: int = 16
    sweep_step_s: float = 2.0
    sweep_replicas: int = 4
    hold_windows_s: tuple[tuple[float, float], ...] = ((14.0, 23.0), (36.0, 45.0))
    horizon_s: float = 10.0


class InTrainingEvaluator:
    """Builds the evaluation environment once and scores a policy on demand.

    Args:
        table_path: Subject table JSON.
        clips_dir: Retargeted clips.
        include: Clip substrings (the training clips; the first is evaluated).
        spec: What to play.
        cfg_kwargs: Task settings forwarded to :func:`~.task.yoga_env_cfg` (action space,
            coordination settings, termination), as in training.
    """

    def __init__(
        self,
        table_path: Path,
        clips_dir: Path,
        include: list[str] | None,
        spec: EvalSpec,
        cfg_kwargs: dict[str, Any],
    ) -> None:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        from myosuite.integrations.yoga import subject_body as sb
        from myosuite.integrations.yoga.task import (
            CTRL_DT,
            SIM_DT,
            clip_names,
            frozen_clip,
            load_yoga_clips,
            start_sweep,
            yoga_env_cfg,
        )

        self.spec = spec
        table = sb.SubjectTable.load(table_path)
        model = sb.compile_spec(sb.build_subject_spec(table), SIM_DT)
        sources = load_yoga_clips(clips_dir, model, include)
        src = sources[0]
        hz = float(src.frequency_hz)
        frozen = tuple(
            frozen_clip(src, min(int(round(t * hz)), len(src.qpos) - 1), spec.horizon_s)
            for t in spec.frozen_at_s
        )
        sweep = start_sweep((src,), spec.sweep_step_s)
        self.bank = frozen + sweep
        names = clip_names(self.bank)
        self.names = [f"frozen@{t:g}s" for t in spec.frozen_at_s] + names[len(frozen) :]
        self.n_frozen = len(frozen)
        self.sweep_start_s = [
            float(n.rsplit("@", 1)[1][:-1]) for n in names[len(frozen) :]
        ]
        reps = [spec.frozen_replicas] * len(frozen) + [spec.sweep_replicas] * len(sweep)
        self.clip_of = np.concatenate([np.full(r, i) for i, r in enumerate(reps)])
        kwargs = dict(cfg_kwargs)
        if kwargs.get("action_space") == "joint" and kwargs.get("feedforward") is None:
            from myosuite.integrations.yoga.feedforward import (
                bank_feedforward,
                coord_options,
            )

            coord = kwargs.get("coord")
            if coord is None or coord.ff_scale != 0.0 or coord.gain_mode != "teacher":
                kwargs["feedforward"] = bank_feedforward(
                    Path(table_path),
                    sources[:1],
                    self.bank,
                    Path(clips_dir).parent / "feedforward",
                    **coord_options(coord),
                )
        cfg = yoga_env_cfg(
            table, self.bank, len(self.clip_of), max_episode_steps=100000, **kwargs
        )
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
        self.wrapped = RslRlVecEnvWrapper(self.env)
        lengths = np.array([len(c.qpos) for c in self.bank])
        self.lengths = np.minimum(lengths, int(round(spec.horizon_s / CTRL_DT)))[
            self.clip_of
        ]
        self.ctrl_dt = CTRL_DT

    def __call__(self, policy: Any) -> dict[str, float]:
        """Mean-action survival on every evaluation clip; returns scalar metrics (s)."""
        import torch

        from myosuite.integrations.yoga.evaluate import _force_starts
        from myosuite.integrations.yoga.task import ENTITY

        n_env = len(self.clip_of)
        alive = np.ones(n_env, bool)
        survived = np.zeros(n_env, int)
        with torch.inference_mode():
            self.wrapped.reset()
            _force_starts(self.env, ENTITY, self.clip_of.tolist(), [0] * n_env)
            obs = self.wrapped.get_observations()
            for t in range(int(self.lengths.max()) - 1):
                obs, _, dones, _ = self.wrapped.step(policy(obs))
                d = dones.cpu().numpy().astype(bool)
                survived[alive & ~d] += 1
                alive &= ~d
                alive &= t + 2 < self.lengths
                if not alive.any():
                    break
        full = survived + 2 >= self.lengths  # survived to the horizon (or the clip end)
        sec = survived * self.ctrl_dt
        out: dict[str, float] = {}
        for i, t in enumerate(self.spec.frozen_at_s):
            k = self.clip_of == i
            out[f"frozen_{t:g}s_mean_s"] = float(sec[k].mean())
            out[f"frozen_{t:g}s_held_frac"] = float(full[k].mean())
        per_start = np.array(
            [
                sec[self.clip_of == self.n_frozen + j].mean()
                for j in range(len(self.sweep_start_s))
            ]
        )
        hold = np.array(
            [
                any(a <= s <= b for a, b in self.spec.hold_windows_s)
                for s in self.sweep_start_s
            ]
        )
        out["sweep_hold_mean_s"] = (
            float(per_start[hold].mean()) if hold.any() else float("nan")
        )
        out["sweep_other_mean_s"] = (
            float(per_start[~hold].mean()) if (~hold).any() else float("nan")
        )
        out["sweep_all_mean_s"] = float(per_start.mean())
        out["sweep_completed_frac"] = float(full[self.clip_of >= self.n_frozen].mean())
        return out

    def close(self) -> None:
        self.env.close()
