"""Evaluation rollouts: fresh episodes, goals counted per episode.

One *episode* is one env from a fresh grasp until it drops the object or the
episode budget runs out. The evaluator counts successes itself from the goal
command term (``success`` per step) rather than trusting any env-side counter,
so the number is the same regardless of the student's config.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EpisodeRecord:
    round_index: int
    env_index: int
    goals_reached: int
    length: int
    dropped: bool
    first_goal_step: int | None
    mean_error_rad: float
    final_error_rad: float


@dataclass
class EvalMetrics:
    episodes: list[EpisodeRecord]
    ctrl_dt: float
    max_steps: int
    success_tolerance_rad: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @property
    def goals_per_episode(self) -> float:
        return self._mean([float(e.goals_reached) for e in self.episodes])

    @property
    def goals_per_episode_std(self) -> float:
        values = [float(e.goals_reached) for e in self.episodes]
        return float(np.std(values)) if values else 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that reached at least one goal."""
        return self._mean([1.0 if e.goals_reached > 0 else 0.0 for e in self.episodes])

    @property
    def drop_rate(self) -> float:
        return self._mean([1.0 if e.dropped else 0.0 for e in self.episodes])

    @property
    def mean_episode_seconds(self) -> float:
        return self._mean([e.length * self.ctrl_dt for e in self.episodes])

    @property
    def mean_error_rad(self) -> float:
        return self._mean([e.mean_error_rad for e in self.episodes])

    @property
    def goals_per_minute(self) -> float:
        total_seconds = sum(e.length for e in self.episodes) * self.ctrl_dt
        if total_seconds <= 0:
            return 0.0
        return 60.0 * sum(e.goals_reached for e in self.episodes) / total_seconds

    def summary(self) -> dict[str, float | int]:
        return {
            "num_episodes": self.num_episodes,
            "goals_per_episode": self.goals_per_episode,
            "goals_per_episode_std": self.goals_per_episode_std,
            "success_rate": self.success_rate,
            "drop_rate": self.drop_rate,
            "mean_episode_seconds": self.mean_episode_seconds,
            "mean_error_rad": self.mean_error_rad,
            "goals_per_minute": self.goals_per_minute,
            "max_steps": self.max_steps,
            "success_tolerance_rad": self.success_tolerance_rad,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "episodes": [asdict(e) for e in self.episodes],
            **({"extra": self.extra} if self.extra else {}),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def episode_budget_steps(env: Any) -> int:
    """Ctrl steps per episode from the env config (the env truncates at this count)."""
    return int(math.ceil(float(env.cfg.max_episode_seconds) / float(env.cfg.ctrl_dt)))


def run_evaluation_episodes(
    env: Any,
    wrapped_env: Any,
    policy: Callable[[Any], torch.Tensor],
    *,
    rounds: int,
    seed: int,
    max_steps: int | None = None,
    command_name: str = "goal",
) -> EvalMetrics:
    """Run ``rounds`` batches of ``env.num_envs`` fresh episodes and record each.

    Each round re-seeds the env's generator and ``init_state`` resets every env
    from a cached grasp; the episode counters are then zeroed so every episode
    runs the full budget unless the object is dropped.
    """
    num_envs = int(env.num_envs)
    budget = int(max_steps) if max_steps is not None else episode_budget_steps(env)
    goal_term = env.command_manager.get_term(command_name)
    records: list[EpisodeRecord] = []
    for round_index in range(int(rounds)):
        seed_everything(int(seed) + round_index)
        env.seed(int(seed) + round_index)
        env.init_state()
        # UniLab's ``init_state`` keeps a random per-env episode offset from the
        # seed stream (it staggers time-outs during training). Left in place it
        # truncates evaluation episodes at a random, policy-independent length,
        # so overwrite the counters through the rsl_rl VecEnv contract, which
        # forwards to ``ManagerBasedRlEnv.set_episode_length_buf``.
        wrapped_env.episode_length_buf = torch.zeros(num_envs, dtype=torch.int64)
        obs = wrapped_env.get_observations()
        active = np.ones(num_envs, dtype=bool)
        goals = np.zeros(num_envs, dtype=np.int64)
        first_goal = np.full(num_envs, -1, dtype=np.int64)
        error_sum = np.zeros(num_envs, dtype=np.float64)
        last_error = np.zeros(num_envs, dtype=np.float64)
        length = np.zeros(num_envs, dtype=np.int64)
        dropped = np.zeros(num_envs, dtype=bool)
        for step in range(budget + 1):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _reward, dones, infos = wrapped_env.step(actions)
            success = goal_term.success.detach().cpu().numpy().astype(bool).reshape(-1)
            error = goal_term.error.detach().cpu().numpy().astype(np.float64).reshape(-1)
            done = dones.detach().cpu().numpy().astype(bool).reshape(-1)
            # Done rows were auto-reset inside the env; the adapter reports which
            # of them timed out, the rest terminated (dropped the object).
            time_outs = infos.get("time_outs") if isinstance(infos, dict) else None
            if isinstance(time_outs, torch.Tensor):
                timed_out = time_outs.detach().cpu().numpy().astype(bool).reshape(-1)
            else:
                timed_out = np.zeros_like(done)
            terminated = done & ~timed_out
            # Rows that finished this step were auto-reset inside the env and
            # their flags were cleared, so only count rows still running.
            counted = active & ~done
            goals[counted] += success[counted]
            error_sum[counted] += error[counted]
            last_error[counted] = error[counted]
            length[active] += 1
            newly_first = counted & success & (first_goal < 0)
            first_goal[newly_first] = step
            finished = active & done
            dropped[finished] = terminated[finished]
            active &= ~done
            if not active.any():
                break
        for env_index in range(num_envs):
            steps_alive = int(length[env_index])
            records.append(
                EpisodeRecord(
                    round_index=round_index,
                    env_index=env_index,
                    goals_reached=int(goals[env_index]),
                    length=steps_alive,
                    dropped=bool(dropped[env_index]),
                    first_goal_step=(
                        int(first_goal[env_index]) if first_goal[env_index] >= 0 else None
                    ),
                    mean_error_rad=float(error_sum[env_index] / max(steps_alive, 1)),
                    final_error_rad=float(last_error[env_index]),
                )
            )
    return EvalMetrics(
        episodes=records,
        ctrl_dt=float(env.cfg.ctrl_dt),
        max_steps=budget,
        success_tolerance_rad=float(goal_term.success_tolerance_rad),
    )


__all__ = [
    "EpisodeRecord",
    "EvalMetrics",
    "episode_budget_steps",
    "run_evaluation_episodes",
    "seed_everything",
]
