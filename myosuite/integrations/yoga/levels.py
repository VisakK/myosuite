# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A discrete policy head for muscle excitations (rsl_rl ``Distribution``).

Each of the first ``output_dim - n_gauss`` outputs (the muscles) is one categorical choice
among ``levels`` evenly spaced excitations in ``[0, 1]``; the last ``n_gauss`` outputs (the
neck motors) are Gaussian with a learnt standard deviation. The choices are independent given
the observation, so the probabilities, and with them the exploration, depend on the state.

Selected in the runner config with
``distribution_cfg={"class_name": LEVELS_CLASS, "levels": 5, "n_gauss": 3, ...}``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution

LEVELS_CLASS = "myosuite.integrations.yoga.levels:LevelsDistribution"
_LOG_2PI = math.log(2.0 * math.pi)


def level_logits(raw: torch.Tensor, ordinal: bool) -> torch.Tensor:
    """Logits over the levels from the network's ``(..., K)`` outputs.

    ``ordinal``: output ``k`` is the log-odds of level ``k`` against level ``k - 1`` (the
    first is unused), so one output moves all the levels above a threshold together.
    """
    if not ordinal:
        return raw
    steps = torch.cat([torch.zeros_like(raw[..., :1]), raw[..., 1:]], dim=-1)
    return steps.cumsum(dim=-1)


def expected_level(logp: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Expected excitation ``(...)`` of ``(..., K)`` log-probabilities."""
    return (logp.exp() * values).sum(dim=-1)


class LevelsDistribution(Distribution):
    """Factorised categorical over excitation levels, plus Gaussian outputs.

    Args:
        output_dim: Action size (categorical outputs first, Gaussian outputs last).
        levels: Number of excitation levels per categorical output.
        n_gauss: Number of Gaussian outputs at the end of the action.
        init_std: Initial standard deviation of the Gaussian outputs.
        max_std: Cap on that standard deviation.
        p_rest: Probability of level 0 at initialisation (the rest is shared evenly).
        deterministic: Output without sampling: ``"mean"`` (the expected excitation) or
            ``"mode"`` (the most likely level).
        ordinal: Parameterise the logits by adjacent log-odds (:func:`level_logits`).
    """

    def __init__(
        self,
        output_dim: int,
        levels: int = 5,
        n_gauss: int = 0,
        init_std: float = 0.2,
        max_std: float = 1.0,
        p_rest: float = 0.9,
        deterministic: str = "mean",
        ordinal: bool = False,
    ) -> None:
        super().__init__(output_dim)
        if levels < 2:
            raise ValueError(f"levels must be at least 2, got {levels}")
        if not 0.0 < p_rest < 1.0:
            raise ValueError(f"p_rest must be in (0, 1), got {p_rest}")
        if deterministic not in ("mean", "mode"):
            raise ValueError(
                f"deterministic must be 'mean' or 'mode', got {deterministic!r}"
            )
        self.k, self.n_cat, self.n_gauss = levels, output_dim - n_gauss, n_gauss
        self.p_rest, self.det, self.ordinal = p_rest, deterministic, ordinal
        self.max_std = max_std
        self.register_buffer("values", torch.linspace(0.0, 1.0, levels))
        self.std_param = nn.Parameter(init_std * torch.ones(n_gauss))
        self._logp: torch.Tensor | None = None
        self._mu: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.n_cat * self.k + self.n_gauss

    def _split(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probabilities ``(..., n_cat, K)`` and Gaussian means ``(..., n_gauss)``."""
        n = self.n_cat * self.k
        raw = out[..., :n].reshape(*out.shape[:-1], self.n_cat, self.k)
        return level_logits(raw, self.ordinal).log_softmax(dim=-1), out[..., n:]

    def update(self, mlp_output: torch.Tensor) -> None:
        self._logp, self._mu = self._split(mlp_output)
        std = self.std_param.clamp(1e-6, self.max_std)
        self._std = std.expand_as(self._mu)

    def sample(self) -> torch.Tensor:
        # Gumbel-max: one categorical draw per muscle without reshaping.
        u = torch.rand_like(self._logp).clamp_(1e-10, 1.0 - 1e-7)
        idx = (self._logp - torch.log(-torch.log(u))).argmax(dim=-1)
        gauss = self._mu + self._std * torch.randn_like(self._mu)
        return torch.cat([self.values[idx], gauss], dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logp, mu = self._split(mlp_output)
        if self.det == "mode":
            return torch.cat([self.values[logp.argmax(dim=-1)], mu], dim=-1)
        return torch.cat([expected_level(logp, self.values), mu], dim=-1)

    def as_deterministic_output_module(self) -> nn.Module:
        return _DeterministicLevels(self)

    @property
    def mean(self) -> torch.Tensor:
        return torch.cat([expected_level(self._logp, self.values), self._mu], dim=-1)

    @property
    def std(self) -> torch.Tensor:
        p = self._logp.exp()
        m = (p * self.values).sum(dim=-1, keepdim=True)
        var = (p * (self.values - m) ** 2).sum(dim=-1)
        return torch.cat([var.sqrt(), self._std], dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        cat = -(self._logp.exp() * self._logp).sum(dim=(-1, -2))
        gauss = (0.5 + 0.5 * _LOG_2PI + self._std.log()).sum(dim=-1)
        return cat + gauss

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._logp.flatten(-2), self._mu, self._std)

    def level_index(self, outputs: torch.Tensor) -> torch.Tensor:
        """Level indices ``(..., n_cat)`` of the categorical part of an action."""
        idx = torch.round(outputs[..., : self.n_cat] * (self.k - 1)).long()
        return idx.clamp(0, self.k - 1)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        idx = self.level_index(outputs).unsqueeze(-1)
        cat = self._logp.gather(-1, idx).squeeze(-1).sum(dim=-1)
        z = (outputs[..., self.n_cat :] - self._mu) / self._std
        gauss = (-0.5 * z * z - self._std.log() - 0.5 * _LOG_2PI).sum(dim=-1)
        return cat + gauss

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        lo, mo, so = old_params
        ln, mn, sn = new_params
        cat = (lo.exp() * (lo - ln)).sum(dim=-1)
        gauss = (sn / so).log() + (so * so + (mo - mn) ** 2) / (2.0 * sn * sn) - 0.5
        return cat + gauss.sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Small output weights; biases that put ``p_rest`` on level 0."""
        last = mlp[-1]
        gap = math.log(self.p_rest * (self.k - 1) / (1.0 - self.p_rest))
        with torch.no_grad():
            last.weight.mul_(0.01)
            last.bias.zero_()
            bias = last.bias[: self.n_cat * self.k].view(self.n_cat, self.k)
            if self.ordinal:
                bias[:, 1] = -gap
            else:
                bias[:, 0] = gap

    def stats(self) -> dict[str, float]:
        """Summary of the last updated distribution (for logging)."""
        if self._logp is None:
            return {}
        with torch.no_grad():
            p = self._logp.exp()
            ent = -(p * self._logp).sum(dim=-1)
            out: dict[str, Any] = {
                "mean_excitation": (p * self.values).sum(dim=-1).mean(),
                "p_rest": p[..., 0].mean(),
                "p_full": p[..., -1].mean(),
                "max_prob": p.max(dim=-1).values.mean(),
                "entropy_per_muscle": ent.mean(),
                "decided_frac": (p.max(dim=-1).values > 0.99).float().mean(),
            }
            if self.n_gauss:
                out["neck_std"] = self._std.mean()
            return {k: float(v) for k, v in out.items()}


class _DeterministicLevels(nn.Module):
    """Export-friendly deterministic output of a :class:`LevelsDistribution`."""

    def __init__(self, dist: LevelsDistribution) -> None:
        super().__init__()
        self.k, self.n_cat = dist.k, dist.n_cat
        self.det, self.ordinal = dist.det, dist.ordinal
        self.register_buffer("values", dist.values.clone())

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        n = self.n_cat * self.k
        raw = out[..., :n].reshape(*out.shape[:-1], self.n_cat, self.k)
        logp = level_logits(raw, self.ordinal).log_softmax(dim=-1)
        if self.det == "mode":
            return torch.cat([self.values[logp.argmax(dim=-1)], out[..., n:]], dim=-1)
        return torch.cat([expected_level(logp, self.values), out[..., n:]], dim=-1)
