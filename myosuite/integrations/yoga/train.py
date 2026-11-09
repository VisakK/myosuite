# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train the muscle-driven yoga tracker with PPO (rsl_rl) on mjlab.

CLI::

    python -m myosuite.integrations.yoga.train --table subject_table.json \
        --clips-dir <retargeted clips> --log-dir <run dir> --num-envs 2048 \
        --iterations 20000 [--resume <run dir>/model_N.pt]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path


def main() -> None:
    """Build the task, then run PPO."""
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
    args = ap.parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

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
    clips = register_yoga_task(
        args.table, args.clips_dir, num_envs=args.num_envs, include=args.include, exclude=args.exclude,
        effort_weight=args.effort_weight, max_site_err=args.max_site_err,
    )  # fmt: skip
    env = ManagerBasedRlEnv(cfg=load_env_cfg(TASK_ID), device="cuda")
    wrapped = RslRlVecEnvWrapper(env)
    runner_cfg = dataclasses.asdict(
        yoga_runner_cfg(
            hidden=tuple(args.hidden), init_std=args.init_std, desired_kl=args.desired_kl, seed=args.seed,
            max_std=args.max_std, entropy_coef=args.entropy_coef,
            logger="tensorboard",
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
            },
            indent=1,
        )
    )
    install_episode_reward_logging_patch()
    runner = MjlabOnPolicyRunner(
        env=wrapped, train_cfg=runner_cfg, log_dir=str(args.log_dir), device="cuda"
    )
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
