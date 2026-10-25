"""Scaled container-management MDP inspired by Pendyala & Glasmachers (2025).

This is an educational simulator, not the authors' original ContainerGym code.
Volumes are normalized to [0, 1]; paper capacity 40 is therefore represented by 1.
"""
from __future__ import annotations
import numpy as np


class WasteSortingEnv:
    """Discrete actions: 0=no-op, i+1=empty container i."""

    LOW_PEAK = 0.35
    HIGH_PEAK = 0.68
    CAPACITY = 1.0
    PEAK_TOLERANCE = 1.0 / 40.0  # paper's +/-1 volume unit, normalized by capacity 40

    def __init__(self, stage: int = 3, seed: int = 0, n_containers: int | None = None,
                 horizon: int | None = None):
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        self.stage = stage
        self.n = n_containers if n_containers is not None else (2 if stage == 1 else 3)
        if not 1 <= self.n <= 3:
            raise ValueError("This fixed-size teaching policy supports 1-3 containers")
        self.horizon = horizon if horizon is not None else (40 if stage == 1 else 60)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.levels = self.rng.uniform(0.08, 0.24, self.n)
        self.busy = 0
        self.t = 0
        self.stats = dict(throughput=0.0, violations=0, collisions=0,
                          energy=0.0, successful_empties=0, peak_deviation=0.0,
                          idle_steps=0, overrides=0)
        return self._obs()

    def _obs(self):
        # Fixed six-dimensional observation enables transfer across curriculum stages.
        levels = np.pad(self.levels, (0, 3 - self.n))
        return np.asarray([*levels, self.busy / 2.0, self.t / self.horizon, 1.0], dtype=np.float32)

    @staticmethod
    def _gaussian_reward(volume: float, peak: float, height: float, width: float = 0.09) -> float:
        return height * np.exp(-0.5 * ((volume - peak) / width) ** 2)

    def _empty_reward(self, volume: float) -> float:
        """Three-phase reward curriculum from the paper (adapted to normalized volume)."""
        r_pen = -0.25
        if self.stage == 1:
            # Paper Algorithm 1: one Gaussian centered on the higher peak.
            gaussian = np.exp(-0.5 * ((volume - self.HIGH_PEAK) / 0.09) ** 2)
            return float(r_pen + (1.0 - r_pen) * gaussian)
        if self.stage == 2:
            # Paper Algorithm 1: add both Gaussian peaks; higher peak has larger height.
            low = np.exp(-0.5 * ((volume - self.LOW_PEAK) / 0.07) ** 2)
            high = np.exp(-0.5 * ((volume - self.HIGH_PEAK) / 0.09) ** 2)
            return float(r_pen + (0.55 - r_pen) * low + (1.0 - r_pen) * high)
        # Paper Algorithm 1: strict step reward within +/-1 physical volume unit.
        if (abs(volume - self.LOW_PEAK) <= self.PEAK_TOLERANCE or
                abs(volume - self.HIGH_PEAK) <= self.PEAK_TOLERANCE):
            return 1.0
        # The paper's Phase 3 is a strict step reward: zero outside either
        # target window. Do not add a distance penalty here; that would change
        # the published reward definition and can bias the policy unexpectedly.
        return 0.0

    def step(self, action: int):
        if not isinstance(action, (int, np.integer)) or action < 0 or action > self.n:
            raise ValueError(f"action must be an integer in [0, {self.n}]")
        reward = 0.0
        # Collision state: multiple containers close to their higher peak while PU is busy.
        near_high = self.levels >= (self.HIGH_PEAK - 0.06)
        if self.busy > 0 and int(near_high.sum()) >= 2:
            self.stats['collisions'] += 1

        if action == 0:
            self.stats['idle_steps'] += int(self.busy == 0)
        elif self.busy > 0:
            # An attempted empty while the PU is busy is lost, as in the paper.
            reward -= 0.25
        else:
            i = action - 1
            volume = float(self.levels[i])
            reward += self._empty_reward(volume)
            self.stats['throughput'] += volume
            self.stats['energy'] += 1.0
            self.stats['successful_empties'] += 1
            self.stats['peak_deviation'] += min(abs(volume - self.LOW_PEAK), abs(volume - self.HIGH_PEAK))
            self.levels[i] = 0.02
            # Processing time is volume-dependent with a small transport overhead.
            self.busy = max(1, int(np.ceil(1.0 + 2.0 * volume)))

        # Curriculum increases stochasticity from phase 1 to phase 3.
        sigma = (0.012, 0.025, 0.035)[self.stage - 1]
        drift = (0.045, 0.05, 0.052)[self.stage - 1]
        self.levels = np.clip(self.levels + self.rng.normal(drift, sigma, self.n), 0.0, None)
        over = self.levels > self.CAPACITY
        count = int(over.sum())
        if count:
            self.stats['violations'] += count
            reward -= 8.0 * count
            self.levels = np.minimum(self.levels, self.CAPACITY)

        self.busy = max(0, self.busy - 1)
        self.t += 1
        done = self.t >= self.horizon or count > 0
        # No extra terminal throughput bonus: the paper's reward is based on
        # target-volume emptying, penalties, and zero-reward no-op actions.
        stats = self.stats.copy()
        stats['press_idle_fraction'] = stats['idle_steps'] / max(self.t, 1)
        stats['mean_peak_deviation'] = stats['peak_deviation'] / max(stats['successful_empties'], 1)
        return self._obs(), float(reward), bool(done), stats
