# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train the muscle-driven yoga tracker with PPO (rsl_rl) on mjlab.

Two action spaces (README §8):

* ``--action-space muscle`` (default, the original task): one excitation per muscle
  through ``sigmoid(5 (a - 0.5))`` plus the three neck torques.
* ``--action-space joint``: PD targets (offsets from the reference), per-limb
  co-contraction and an optional torque residual, turned into muscle excitations every
  physics substep by the muscle coordination layer (:mod:`.coordination`), with static
  feed-forward torques (:mod:`.feedforward`).

``--logger wandb`` logs to Weights & Biases (project ``--wandb-project``; the run is named
after ``--log-dir``). ``--eval-interval N`` scores the mean action every N iterations on
frozen holds and a start sweep (:mod:`.train_eval`) and logs them under ``Eval/``; with the
joint action space the layer's diagnostics are logged under ``MCL/`` every iteration.

CLI::

    python -m myosuite.integrations.yoga.train --table subject_table.json \
        --clips-dir <retargeted clips> --log-dir <run dir> --num-envs 2048 \
        --iterations 20000 [--resume <run dir>/model_N.pt]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any


def _coord_from_args(args: argparse.Namespace) -> Any:
    from myosuite.integrations.yoga.coordination import CoordinationCfg

    return CoordinationCfg(
        dq_scale=args.dq_scale, gain_mode=args.gain_mode, omega_max=args.omega_max,
        zeta=args.zeta, kp_scale=args.kp_scale, cocon_base=args.cocon_base,
        cocon_scale=args.cocon_scale, cocon_max=args.cocon_max, ff_scale=args.ff_scale,
        tau_res_scale=args.tau_res_scale, residual_scale=args.residual_scale,
        horizon_s=args.horizon_s, every_substep=not args.solve_per_step,
        ff_limits=args.ff_limits, planted_inertia=args.planted_inertia,
    )  # fmt: skip


def task_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Task settings of a run (also rebuilt from ``config.json`` by the evaluators)."""
    kw: dict[str, Any] = dict(
        effort_weight=args.effort_weight, max_site_err=args.max_site_err
    )
    if args.action_space == "joint":
        kw.update(action_space="joint", coord=_coord_from_args(args))
    return kw


def build_parser() -> argparse.ArgumentParser:
    """The training CLI (shared with the evaluators, which re-parse ``config.json``)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--steps-per-env", type=int, default=24)
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--include", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-std", type=float, default=0.6)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--effort-weight", type=float, default=1.0)
    ap.add_argument("--max-site-err", type=float, default=0.5)
    ap.add_argument(
        "--desired-kl",
        type=float,
        default=0.3,
        help="adaptive-LR KL target (summed over actions)",
    )
    ap.add_argument(
        "--max-std", type=float, default=None, help="cap on the action noise std"
    )
    ap.add_argument("--entropy-coef", type=float, default=0.001)
    ap.add_argument("--learning-epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    # Action space and the muscle coordination layer.
    ap.add_argument("--action-space", choices=("muscle", "joint"), default="muscle")
    ap.add_argument(
        "--dq-scale",
        type=float,
        default=0.3,
        help="joint: target offset per unit action (rad)",
    )
    ap.add_argument(
        "--gain-mode", choices=("bounded", "inertia", "teacher"), default="bounded"
    )
    ap.add_argument(
        "--omega-max",
        type=float,
        default=80.0,
        help="joint/bounded: highest hinge natural frequency",
    )
    ap.add_argument("--zeta", type=float, default=1.0)
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--cocon-base", type=float, default=0.03)
    ap.add_argument("--cocon-scale", type=float, default=0.1)
    ap.add_argument("--cocon-max", type=float, default=0.4)
    ap.add_argument(
        "--ff-scale",
        type=float,
        default=1.0,
        help="joint: static feed-forward torque scale",
    )
    ap.add_argument(
        "--tau-res-scale",
        type=float,
        default=0.25,
        help="joint: torque residual per unit action (x capacity)",
    )
    ap.add_argument(
        "--residual-scale",
        type=float,
        default=0.0,
        help="joint: per-muscle excitation residual (0: off)",
    )
    ap.add_argument("--horizon-s", type=float, default=0.01)
    ap.add_argument(
        "--solve-per-step",
        action="store_true",
        help="joint: solve once per control step (default: every substep)",
    )
    ap.add_argument(
        "--ff-limits",
        choices=("all", "stiff"),
        default="all",
        help="joint: feed-forward LP limit reactions on every hinge at a limit (round 2) or only on stiff end-ranges",
    )
    ap.add_argument(
        "--planted-inertia",
        action="store_true",
        help="joint: contact-aware hinge inertia (planted distal segment) for the gain cap",
    )
    # Logging and in-training evaluation.
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    ap.add_argument("--wandb-project", default="yoga_muscle")
    ap.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="evaluate every N iterations (0: off)",
    )
    ap.add_argument("--eval-frozen-s", type=float, nargs="*", default=[19.88])
    return ap


def make_runner_class(base: type) -> type:
    """A runner that also logs the coordination layer's diagnostics and in-training scores."""

    class YogaRunner(base):  # type: ignore[misc, valid-type]
        def __init__(
            self, *a: Any, evaluator: Any = None, eval_interval: int = 0, **kw: Any
        ) -> None:
            super().__init__(*a, **kw)
            self._evaluator = evaluator
            self._eval_interval = eval_interval
            orig_log = self.logger.log

            def log(**kwargs: Any) -> None:
                orig_log(**kwargs)
                self._extra_logs(int(kwargs["it"]))

            self.logger.log = log  # type: ignore[method-assign]

        def _extra_logs(self, it: int) -> None:
            writer = self.logger.writer
            if writer is None:
                return
            am = self.env.unwrapped.action_manager
            if "joint_pd" in am.active_terms:
                for k, v in am.get_term("joint_pd").diagnostics().items():
                    writer.add_scalar(f"MCL/{k}", v, it)
            if (
                self._evaluator is not None
                and self._eval_interval > 0
                and it % self._eval_interval == 0
            ):
                t0 = time.time()
                scores = self._evaluator(self.alg.get_policy())
                for k, v in scores.items():
                    writer.add_scalar(f"Eval/{k}", v, it)
                writer.add_scalar("Eval/wall_s", time.time() - t0, it)
                line = " ".join(f"{k} {v:.3f}" for k, v in scores.items())
                print(f"[eval it {it}] {line}", flush=True)
                if self.logger.log_dir is not None:
                    with open(Path(self.logger.log_dir) / "eval.jsonl", "a") as f:
                        f.write(json.dumps({"it": it, **scores}) + "\n")

    return YogaRunner


def main() -> None:
    """Build the task, then run PPO."""
    args = build_parser().parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    import myosuite
    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import (
        install_episode_reward_logging_patch,
    )
    from myosuite.integrations.yoga.task import (
        TASK_ID,
        clip_names,
        register_yoga_task,
        yoga_runner_cfg,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    kw = task_kwargs(args)
    clips = register_yoga_task(
        args.table, args.clips_dir, num_envs=args.num_envs, include=args.include, exclude=args.exclude, **kw,
    )  # fmt: skip
    env = ManagerBasedRlEnv(cfg=load_env_cfg(TASK_ID), device="cuda")
    wrapped = RslRlVecEnvWrapper(env)
    runner_cfg = dataclasses.asdict(
        yoga_runner_cfg(
            hidden=tuple(args.hidden), init_std=args.init_std, desired_kl=args.desired_kl, seed=args.seed,
            max_std=args.max_std, entropy_coef=args.entropy_coef,
            num_learning_epochs=args.learning_epochs, num_mini_batches=args.mini_batches,
            logger=args.logger, wandb_project=args.wandb_project,
            experiment_name="yoga_muscle", num_steps_per_env=args.steps_per_env, max_iterations=args.iterations,
            save_interval=args.save_interval, upload_model=False,
        )
    )  # fmt: skip
    (args.log_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "clips": clip_names(clips),
                "runner": runner_cfg,
                "coord": dataclasses.asdict(kw["coord"]) if "coord" in kw else None,
            },
            indent=1,
        )
    )
    evaluator = None
    if args.eval_interval > 0:
        from myosuite.integrations.yoga.train_eval import EvalSpec, InTrainingEvaluator

        evaluator = InTrainingEvaluator(
            args.table, args.clips_dir, args.include,
            EvalSpec(frozen_at_s=tuple(args.eval_frozen_s)), kw,
        )  # fmt: skip
    install_episode_reward_logging_patch()
    runner = make_runner_class(MjlabOnPolicyRunner)(
        env=wrapped, train_cfg=runner_cfg, log_dir=str(args.log_dir), device="cuda",
        evaluator=evaluator, eval_interval=args.eval_interval,
    )  # fmt: skip
    runner.add_git_repo_to_log(myosuite.__file__)
    if args.resume is not None:
        # Not under inference_mode: the optimizer state must stay trainable.
        runner.load(str(args.resume), map_location="cuda")
        dist = runner.alg.get_policy().distribution
        if args.max_std is not None and dist.std_type == "scalar":
            # A parameter above the clamp gets no gradient; start it at the cap instead.
            with torch.no_grad():
                dist.std_param.clamp_(max=args.max_std)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
