# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Episode starts weighted by where episodes fail (reference-state initialisation).

With uniform starts a passage the policy fails in ends its episodes early and is met less
often than the passages it has learnt. :class:`FailureWeightedStarts` estimates, per start
bin of every clip, how often an episode that passes the bin ends early in it, and draws
part of the starts in proportion.

BeyondMimic and mjlab's tracking task weight by the number of failures per bin. Here the
weight is the failure rate per visit: with capped episodes a late passage is reached from
many starts and an early one from few, so counts would favour the end of a long clip.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class AdaptiveStartsCfg:
    """Settings of :class:`FailureWeightedStarts`.

    Attributes:
        uniform: Share of the starts that stays uniform.
        alpha: Weight of one control step in the moving averages.
        lead: Bins between a failing bin and the nearest start it weights. A start
            inside the failing bin meets the failure from the reference state, not as
            the policy arrives at it.
        kernel: Starts a failure rate is spread over, going back from ``lead``.
        decay: Weight of each earlier start relative to the next one.
        min_fail_s: Episodes that end early sooner than this after their start are left
            out (a start state that cannot be survived says nothing about the policy).
            Episodes from a clip's first frame always count.
        min_hazard: Expected early ends along one clip at which half of the weighted
            share is in use; with fewer the starts return to uniform.
        prior_visits: Visits per averaging window below which a bin's rate is damped.
    """

    uniform: float = 0.3
    alpha: float = 0.001
    lead: int = 1
    kernel: int = 3
    decay: float = 0.8
    min_fail_s: float = 0.75
    min_hazard: float = 0.05
    prior_visits: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.uniform <= 1.0:
            raise ValueError(f"uniform must be in [0, 1], got {self.uniform}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {self.kernel}")
        if self.lead < 0:
            raise ValueError(f"lead must not be negative, got {self.lead}")


class FailureWeightedStarts:
    """Start bins of a clip bank, drawn in proportion to the failure rates near them.

    A bin is ``bin_steps`` frames of one clip; a start is the first frame of a bin. Per
    bin two moving averages are kept, in episodes per control step: the episodes that
    were in the bin (visits) and those that ended early in it (failures). Their ratio is
    the bin's failure rate. A start whose own episodes end within ``min_fail_s`` gets
    less of the weighted share, in proportion (a start state that cannot be survived).

    Args:
        clip_lengths: Frames per clip.
        bin_steps: Frames per bin.
        ctrl_dt: Control step (s).
        num_envs: Number of environments.
        cfg: Settings.
        device: Torch device of the statistics.
    """

    def __init__(
        self,
        clip_lengths: Sequence[int],
        bin_steps: int,
        ctrl_dt: float,
        num_envs: int,
        cfg: AdaptiveStartsCfg,
        device: Any,
    ) -> None:
        import torch

        if bin_steps < 1:
            raise ValueError(f"bin_steps must be at least 1, got {bin_steps}")
        self.cfg = cfg
        self.bin_steps = int(bin_steps)
        self.ctrl_dt = float(ctrl_dt)
        self.num_envs = int(num_envs)
        self.min_fail_steps = int(round(cfg.min_fail_s / ctrl_dt))
        n_bins = torch.tensor(
            [-(-int(n) // self.bin_steps) for n in clip_lengths], device=device
        )
        self.n_bins = n_bins
        self.first = torch.cumsum(n_bins, 0) - n_bins
        clips = torch.arange(len(n_bins), device=device)
        self.clip_of = torch.repeat_interleave(clips, n_bins)
        flat = torch.arange(int(n_bins.sum()), device=device)
        self.bin_in_clip = flat - self.first[self.clip_of]
        # The plain rule: a clip uniformly, then one of its bins uniformly.
        self.uniform_p = 1.0 / (len(n_bins) * n_bins[self.clip_of].float())
        weights = torch.tensor([cfg.decay**j for j in range(cfg.kernel)], device=device)
        self.kernel = weights / weights.sum()
        self.fail = torch.zeros(len(flat), device=device)
        self.visit = torch.zeros(len(flat), device=device)
        self.started = torch.zeros(len(flat), device=device)
        self.died_early = torch.zeros(len(flat), device=device)
        self.skip = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._last_step: int | None = None

    def flat_bin(self, clip_ids: Any, frames: Any) -> Any:
        """Flat bin index of *frames* of clips *clip_ids* (clamped into the clip)."""
        import torch

        b = torch.div(frames.clamp_min(0), self.bin_steps, rounding_mode="floor")
        return self.first[clip_ids] + torch.minimum(b, self.n_bins[clip_ids] - 1)

    def skip_running_episodes(self) -> None:
        """Leave the episode in progress of every environment out of the statistics."""
        import torch

        self.skip = torch.ones_like(self.skip)

    def observe(
        self,
        step_count: int,
        env_ids: Any,
        clip_ids: Any,
        start_frames: Any,
        end_frames: Any,
        failed: Any,
    ) -> None:
        """Count the episodes that ended at one control step.

        Args:
            step_count: The environment's control-step counter.
            env_ids: ``(E,)`` environments whose episodes ended.
            clip_ids: ``(E,)`` their clips.
            start_frames: ``(E,)`` clip frames they started at.
            end_frames: ``(E,)`` clip frames they ended at.
            failed: ``(E,)`` bool, ended early (not by a time limit).
        """
        import torch

        alpha = self.cfg.alpha
        if self._last_step is not None and step_count > self._last_step:
            keep = (1.0 - alpha) ** (step_count - self._last_step)
            self.fail, self.visit = self.fail * keep, self.visit * keep
            self.started, self.died_early = self.started * keep, self.died_early * keep
        self._last_step = step_count
        steps = end_frames - start_frames
        use = (steps > 0) & ~self.skip[env_ids]
        self.skip = self.skip.index_fill(0, env_ids, False)
        if not bool(use.any()):
            return
        # Episodes that ended early right after a start inside the clip count against
        # that start, not against the policy.
        early = use & failed & (steps < self.min_fail_steps) & (start_frames > 0)
        n_bins = len(self.fail)
        began = self.flat_bin(clip_ids[use], start_frames[use])
        self.started = self.started + alpha * torch.bincount(began, minlength=n_bins)
        gone = self.flat_bin(clip_ids[early], start_frames[early])
        self.died_early = self.died_early + alpha * torch.bincount(
            gone, minlength=n_bins
        )
        use = use & ~early
        if not bool(use.any()):
            return
        clips, ended = clip_ids[use], failed[use]
        first_bin = self.flat_bin(clips, start_frames[use])
        # A time limit ends an episode on the frame after the last one it lived.
        last_bin = self.flat_bin(clips, end_frames[use] - (~ended).long())
        # One visit to every bin from the first to the last: a difference array with a
        # spare slot after each clip.
        ones = torch.ones(len(clips), device=self.fail.device)
        diff = torch.zeros(len(self.fail) + len(self.n_bins), device=self.fail.device)
        diff = diff.index_add(0, first_bin + clips, ones)
        diff = diff.index_add(0, last_bin + clips + 1, -ones)
        slots = torch.arange(len(self.fail), device=self.fail.device) + self.clip_of
        self.visit = self.visit + alpha * torch.cumsum(diff, 0)[slots]
        counts = torch.bincount(last_bin[ended], minlength=len(self.fail))
        self.fail = self.fail + alpha * counts.float()

    def hazard(self) -> Any:
        """Per bin: share of the episodes in it that end early there, ``(B,)``."""
        visits = self.visit + self.cfg.prior_visits * self.cfg.alpha
        return (self.fail / visits.clamp_min(1e-12)).clamp(0.0, 1.0)

    def start_quality(self) -> Any:
        """Per start: share of its episodes that outlast ``min_fail_s``, ``(B,)``."""
        started = self.started + self.cfg.prior_visits * self.cfg.alpha
        return (1.0 - self.died_early / started.clamp_min(1e-12)).clamp(0.0, 1.0)

    def weighted_share(self) -> float:
        """Share of the starts drawn by failure rates (the rest is uniform)."""
        per_clip = float(self.hazard().sum()) / len(self.n_bins)
        return (1.0 - self.cfg.uniform) * per_clip / (per_clip + self.cfg.min_hazard)

    def start_probabilities(self) -> Any:
        """Probability of every flat start bin, ``(B,)``, summing to 1."""
        import torch

        hazard = self.hazard()
        spread = torch.zeros_like(hazard)
        flat = torch.arange(len(hazard), device=hazard.device)
        for j, w in enumerate(self.kernel):
            # The rate of bin b weights the start lead + j bins before it, in its clip.
            target = torch.maximum(flat - self.cfg.lead - j, self.first[self.clip_of])
            spread = spread.index_add(0, target, w * hazard)
        spread = spread * self.start_quality()
        total = spread.sum()
        if float(total) <= 0.0:
            return self.uniform_p
        share = self.weighted_share()
        return share * spread / total + (1.0 - share) * self.uniform_p

    def sample(self, n: int) -> tuple[Any, Any]:
        """Clip indices and start frames (the first frame of a bin) of *n* starts."""
        import torch

        flat = torch.multinomial(self.start_probabilities(), n, replacement=True)
        return self.clip_of[flat], self.bin_in_clip[flat] * self.bin_steps

    def clip_pass_estimate(self) -> Any:
        """Per clip: chance of passing every bin, from the bins' failure rates, ``(C,)``."""
        import torch

        log_pass = torch.log1p(-self.hazard().clamp(max=1.0 - 1e-6))
        total = torch.zeros(len(self.n_bins), device=log_pass.device)
        return total.index_add(0, self.clip_of, log_pass).exp()

    def stats(self) -> dict[str, float]:
        """Summary for logging."""
        p = self.start_probabilities()
        top = int(p.argmax())
        entropy = float(-(p * (p + 1e-12).log()).sum())
        return {
            "weighted_share": self.weighted_share(),
            "top_prob": float(p[top]),
            "top_start_s": float(self.bin_in_clip[top]) * self.bin_steps * self.ctrl_dt,
            "entropy": entropy / math.log(len(p)) if len(p) > 1 else 1.0,
            "falls_per_env_s": float(self.fail.sum()) / self.num_envs / self.ctrl_dt,
            "clip_pass_estimate": float(self.clip_pass_estimate().mean()),
        }
