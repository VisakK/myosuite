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

``--policy-head levels`` (muscle action space) makes the policy discrete: one choice among
``--levels`` excitation levels per muscle (:mod:`.levels`). ``--obs body`` replaces the
world-frame observation by the body-relative one of :mod:`.body_obs`. ``--neck servo`` takes
the neck motors out of the policy (:mod:`.neck`). ``--adaptive-starts`` draws part of the
episode starts in proportion to where episodes end early.

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
import os
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
    from myosuite.integrations.yoga.task import BodyObsCfg

    kw: dict[str, Any] = dict(
        effort_weight=args.effort_weight, max_site_err=args.max_site_err
    )
    if args.action_space == "joint":
        if args.policy_head == "levels":
            raise ValueError("the levels head writes muscle excitations: --action-space muscle")  # fmt: skip
        kw.update(action_space="joint", coord=_coord_from_args(args))
    elif args.policy_head == "levels":
        kw["muscle_mode"] = "excitation"
    if args.neck == "servo":
        kw["neck"] = "servo"
    if args.obs == "body":
        kw["obs"] = BodyObsCfg(
            tuple(args.target_steps), tuple(args.preview_steps), args.muscle_state
        )
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
    ap.add_argument(
        "--critic-hidden",
        type=int,
        nargs="+",
        default=None,
        help="critic layer sizes (default: as --hidden)",
    )
    ap.add_argument("--effort-weight", type=float, default=1.0)
    ap.add_argument("--max-site-err", type=float, default=0.5)
    ap.add_argument(
        "--start-interval-s",
        type=float,
        default=None,
        help="start episodes only at multiples of this clip time (default: any frame)",
    )
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
    ap.add_argument(
        "--std-from",
        type=Path,
        default=None,
        help="take the action noise std from this checkpoint (applied after --resume)",
    )
    ap.add_argument(
        "--std-scale", type=float, default=1.0, help="multiply the action noise std"
    )
    ap.add_argument(
        "--freeze-std", action="store_true", help="do not learn the action noise std"
    )
    ap.add_argument("--learning-epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    # Policy head and observation.
    ap.add_argument(
        "--policy-head",
        choices=("gaussian", "levels"),
        default="gaussian",
        help="levels: one categorical choice among excitation levels per muscle",
    )
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument(
        "--p-rest", type=float, default=0.9, help="levels: start probability of level 0"
    )
    ap.add_argument("--levels-output", choices=("mean", "mode"), default="mean")
    ap.add_argument("--ordinal", action="store_true")
    ap.add_argument(
        "--obs",
        choices=("world", "body"),
        default="world",
        help="body: body-relative observation in the ProtoMotions style",
    )
    ap.add_argument("--target-steps", type=int, nargs="+", default=[1])
    ap.add_argument("--preview-steps", type=int, nargs="*", default=[5, 10, 20, 50, 100])  # fmt: skip
    ap.add_argument("--muscle-state", choices=("full", "act", "none"), default="full")
    ap.add_argument(
        "--neck",
        choices=("policy", "servo"),
        default="policy",
        help="muscle action space: servo holds the neck on the reference (no policy output)",
    )
    # Episode starts.
    ap.add_argument(
        "--adaptive-starts",
        action="store_true",
        help="draw part of the starts in proportion to where episodes end early",
    )
    ap.add_argument("--adaptive-uniform", type=float, default=0.3)
    ap.add_argument("--adaptive-lead", type=int, default=1)
    ap.add_argument("--adaptive-kernel", type=int, default=3)
    ap.add_argument("--adaptive-decay", type=float, default=0.8)
    ap.add_argument("--adaptive-alpha", type=float, default=0.001)
    ap.add_argument("--adaptive-min-fail-s", type=float, default=0.75)
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
    ap.add_argument(
        "--eval-whole",
        type=int,
        default=0,
        help="whole-clip rollouts from the first frame per evaluation (0: none)",
    )
    ap.add_argument(
        "--eval-also",
        nargs="*",
        choices=("sampled", "mode"),
        default=[],
        help="score these controllers too (frozen holds and sweep), under Eval_<name>/",
    )
    ap.add_argument(
        "--eval-extra-every",
        type=int,
        default=1,
        help="play the whole clip and the --eval-also controllers only at every N-th evaluation",
    )
    return ap


def make_runner_class(base: type) -> type:
    """A runner that also logs the coordination layer's diagnostics and in-training scores."""

    class YogaRunner(base):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *a: Any,
            evaluator: Any = None,
            eval_interval: int = 0,
            eval_also: tuple[str, ...] = (),
            eval_extra_every: int = 1,
            **kw: Any,
        ) -> None:
            super().__init__(*a, **kw)
            self._evaluator = evaluator
            self._eval_interval = eval_interval
            self._eval_also = tuple(eval_also)
            self._eval_extra_every = max(1, eval_extra_every)
            orig_log = self.logger.log

            def log(**kwargs: Any) -> None:
                orig_log(**kwargs)
                self._extra_logs(int(kwargs["it"]))

            self.logger.log = log  # type: ignore[method-assign]

        def _log_starts(self, writer: Any, it: int) -> None:
            """The start sampler's summary, and its start probabilities at evaluations."""
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                _resolve_mimic_mjlab_ids,
            )
            from myosuite.integrations.yoga.task import ENTITY

            cache = _resolve_mimic_mjlab_ids(self.env.unwrapped, ENTITY, "fullbody")
            sampler = cache.get("start_sampler")
            if sampler is None:
                return
            for k, v in sampler.stats().items():
                writer.add_scalar(f"Starts/{k}", v, it)
            every = self._eval_interval or 50
            if it % every == 0 and self.logger.log_dir is not None:
                visit = sampler.visit / sampler.visit.sum().clamp_min(1e-12)
                row = {"it": it, "bin_s": sampler.bin_steps * sampler.ctrl_dt}
                for key, values in (
                    ("p", sampler.start_probabilities()),
                    ("failure_rate", sampler.hazard()),
                    ("sample_share", visit),
                    ("start_quality", sampler.start_quality()),
                ):
                    row[key] = [round(x, 5) for x in values.tolist()]
                with open(Path(self.logger.log_dir) / "starts.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")

        def _extra_logs(self, it: int) -> None:
            writer = self.logger.writer
            if writer is None:
                return
            am = self.env.unwrapped.action_manager
            if "joint_pd" in am.active_terms:
                for k, v in am.get_term("joint_pd").diagnostics().items():
                    writer.add_scalar(f"MCL/{k}", v, it)
            if "neck" in am.active_terms and hasattr(am.get_term("neck"), "tracking_error"):  # fmt: skip
                err = am.get_term("neck").tracking_error()
                writer.add_scalar("Neck/servo_error_rms_deg", float(err.square().mean().sqrt()) * 57.2958, it)  # fmt: skip
            self._log_starts(writer, it)
            policy = self.alg.get_policy()
            dist = policy.distribution
            if hasattr(dist, "stats"):
                for k, v in dist.stats().items():
                    writer.add_scalar(f"Levels/{k}", v, it)
            if (
                self._evaluator is not None
                and self._eval_interval > 0
                and it % self._eval_interval == 0
            ):
                t0 = time.time()
                extra = (it // self._eval_interval) % self._eval_extra_every == 0
                scores = self._evaluator(policy, whole=extra)
                for k, v in scores.items():
                    writer.add_scalar(f"Eval/{k}", v, it)
                row = {"it": it, **scores}
                for name in self._eval_also if extra else ():
                    if name == "mode" and not hasattr(dist, "det"):
                        continue
                    kept = getattr(dist, "det", None)
                    if name == "mode":
                        dist.det = "mode"
                    try:
                        other = self._evaluator(
                            policy, stochastic=name == "sampled", whole=False
                        )
                    finally:
                        if name == "mode":
                            dist.det = kept
                    for k, v in other.items():
                        writer.add_scalar(f"Eval_{name}/{k}", v, it)
                    row[name] = other
                writer.add_scalar("Eval/wall_s", time.time() - t0, it)
                line = " ".join(f"{k} {v:.3f}" for k, v in scores.items())
                print(f"[eval it {it}] {line}", flush=True)
                if self.logger.log_dir is not None:
                    with open(Path(self.logger.log_dir) / "eval.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")

    return YogaRunner


def set_action_noise(
    dist: Any,
    std_from: Path | None = None,
    scale: float = 1.0,
    max_std: float | None = None,
    freeze: bool = False,
) -> None:
    """Set, scale, cap and optionally freeze a scalar action noise std in place.

    Args:
        dist: The actor's rsl_rl ``GaussianDistribution`` (``std_type="scalar"``).
        std_from: Checkpoint whose std is copied first (``None``: keep the present one).
        scale: Factor applied to the std.
        max_std: Cap; a parameter above the clamp would get no gradient.
        freeze: Stop learning the std.
    """
    import torch

    if dist.std_type != "scalar":
        raise ValueError("the action noise options need a scalar std")
    with torch.no_grad():
        if std_from is not None:
            ckpt = torch.load(std_from, map_location="cpu", weights_only=False)
            src = ckpt["actor_state_dict"]["distribution.std_param"]
            dist.std_param.copy_(src.to(dist.std_param))
        dist.std_param.mul_(scale)
        if max_std is not None:
            dist.std_param.clamp_(max=max_std)
    if freeze:
        dist.std_param.requires_grad_(False)


def stagger_episodes(env: Any, entity: str) -> None:
    """Spread the episode clocks over one episode without moving the reference.

    rsl_rl's ``init_at_random_ep_len`` moves only the clocks. The clip frame is start +
    clock, so the reference would jump ahead of the body for the first rollout. Here the
    start is moved back by the same amount: the frame stays, the episode ends early.
    """
    import torch

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _resolve_mimic_mjlab_ids,
    )

    cache = _resolve_mimic_mjlab_ids(env, entity, "fullbody")
    src = cache["clip_source"]
    shift = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    env.episode_length_buf[:] = shift
    src._start_offsets -= shift
    src._last_step[:] = shift
    cache["synced_inputs"] = None
    if cache.get("start_sampler") is not None:
        # The shifted starts are not where these episodes began.
        cache["start_sampler"].skip_running_episodes()


def keep_wandb_config_on_resume() -> None:
    """Let a W&B run that is resumed by id keep the config it has.

    rsl_rl stores the whole env config at start, and W&B refuses values that changed on a
    resumed run. The env config holds function addresses, which differ in every process.
    """
    from rsl_rl.utils.wandb_utils import WandbSummaryWriter
    from wandb.sdk.lib.config_util import ConfigError

    store = WandbSummaryWriter.store_config

    def store_config(self: Any, env_cfg: Any, train_cfg: dict) -> None:
        try:
            store(self, env_cfg, train_cfg)
        except ConfigError:
            print("[wandb] resumed run: the stored config is kept", flush=True)

    WandbSummaryWriter.store_config = store_config  # type: ignore[method-assign]


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
        ENTITY,
        TASK_ID,
        LevelsCfg,
        clip_names,
        register_yoga_task,
        yoga_runner_cfg,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    kw = task_kwargs(args)
    adaptive = None
    if args.adaptive_starts:
        from myosuite.envs.myo.backends.mjlab.start_sampling import AdaptiveStartsCfg

        adaptive = AdaptiveStartsCfg(
            uniform=args.adaptive_uniform, alpha=args.adaptive_alpha, lead=args.adaptive_lead, kernel=args.adaptive_kernel,
            decay=args.adaptive_decay, min_fail_s=args.adaptive_min_fail_s,
        )  # fmt: skip
    clips = register_yoga_task(
        args.table, args.clips_dir, num_envs=args.num_envs, include=args.include, exclude=args.exclude,
        start_interval_s=args.start_interval_s, adaptive_starts=adaptive, **kw,
    )  # fmt: skip
    env = ManagerBasedRlEnv(cfg=load_env_cfg(TASK_ID), device="cuda")
    wrapped = RslRlVecEnvWrapper(env)
    levels = None
    n_gauss = 0
    if args.policy_head == "levels":
        levels = LevelsCfg(args.levels, args.p_rest, args.levels_output, args.ordinal)
        am = env.action_manager
        n_gauss = am.total_action_dim - am.get_term("muscles").action_dim
    runner_cfg = dataclasses.asdict(
        yoga_runner_cfg(
            hidden=tuple(args.hidden), init_std=args.init_std, desired_kl=args.desired_kl, seed=args.seed,
            max_std=args.max_std, entropy_coef=args.entropy_coef,
            num_learning_epochs=args.learning_epochs, num_mini_batches=args.mini_batches,
            levels=levels, n_gauss=n_gauss, body_obs=args.obs == "body",
            critic_hidden=None if args.critic_hidden is None else tuple(args.critic_hidden),
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
                "obs": dataclasses.asdict(kw["obs"]) if "obs" in kw else None,
                "muscle_mode": kw.get("muscle_mode", "sigmoid"),
                "neck": kw.get("neck", "policy"),
            },
            indent=1,
        )
    )
    evaluator = None
    if args.eval_interval > 0:
        from myosuite.integrations.yoga.train_eval import EvalSpec, InTrainingEvaluator

        evaluator = InTrainingEvaluator(
            args.table, args.clips_dir, args.include,
            EvalSpec(frozen_at_s=tuple(args.eval_frozen_s), whole_replicas=args.eval_whole), kw,
        )  # fmt: skip
    install_episode_reward_logging_patch()
    runner = make_runner_class(MjlabOnPolicyRunner)(
        env=wrapped, train_cfg=runner_cfg, log_dir=str(args.log_dir), device="cuda",
        evaluator=evaluator, eval_interval=args.eval_interval,
        eval_also=tuple(args.eval_also), eval_extra_every=args.eval_extra_every,
    )  # fmt: skip
    runner.add_git_repo_to_log(myosuite.__file__)
    if args.resume is not None:
        # Not under inference_mode: the optimizer state must stay trainable.
        runner.load(str(args.resume), map_location="cuda")
        # The adaptive schedule's rate is saved only inside the optimizer.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
    dist = runner.alg.get_policy().distribution
    if getattr(dist, "std_type", None) == "scalar":
        set_action_noise(
            dist, args.std_from, args.std_scale, args.max_std, args.freeze_std
        )
    with torch.inference_mode():
        stagger_episodes(env, ENTITY)
    if args.logger == "wandb" and os.environ.get("WANDB_RESUME"):
        keep_wandb_config_on_resume()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
