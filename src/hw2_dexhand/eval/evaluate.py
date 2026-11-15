"""Evaluate one checkpoint on one cell, or a whole submission manifest.

A submission manifest is a YAML file::

    team: team-00
    members: [A, B]
    config: my_config.yaml          # overlay used for every cell (optional)
    cells:
      yaw_absolute/ball: runs/ball_yaw            # run dir -> highest model_*.pt
      so3_relative/cube:
        checkpoint: runs/cube_so3/model_399.pt
        config: cube_config.yaml                  # per-cell overlay (optional)

Paths are relative to the manifest. Each cell is evaluated on the evaluator's
config with the student's observation/action/algo sections, for
``episodes`` fresh episodes in batches of ``num_envs``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hw2_dexhand.eval.challenges import Challenge
from hw2_dexhand.eval.config import StudentConfigError, compose_evaluation, create_env
from hw2_dexhand.eval.policy import CheckpointError, load_policy, resolve_checkpoint
from hw2_dexhand.eval.rollout import EvalMetrics, run_evaluation_episodes
from hw2_dexhand.eval.scoring import CellScore, Rubric, score_cell
from hw2_dexhand.eval.tasks import TaskSpec, parse_cell


@dataclass
class CellResult:
    cell: str
    checkpoint: str | None
    config: str | None
    metrics: EvalMetrics | None
    score: CellScore
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    parts: list[CellResult] = field(default_factory=list)
    challenges: list[CellResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "checkpoint": self.checkpoint,
            "config": self.config,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "score": self.score.to_dict(),
            "warnings": list(self.warnings),
            "error": self.error,
            "seconds": self.seconds,
            **({"parts": [part.to_dict() for part in self.parts]} if self.parts else {}),
            **({"challenges": [c.to_dict() for c in self.challenges]} if self.challenges else {}),
        }


def evaluate_cell(
    spec: TaskSpec,
    checkpoint: str | Path,
    *,
    config: str | Path | None = None,
    episodes: int,
    num_envs: int = 16,
    seed: int = 0,
    device: str = "cpu",
    rubric: Rubric,
    root_dir: str | Path = ".",
    log=print,
    challenge: Challenge | None = None,
    challenges: Sequence[Challenge] = (),
    allow_privileged: bool = False,
) -> CellResult:
    """Evaluate ``checkpoint`` on ``spec``; never raises for a bad submission.

    A multi-object cell (``multi/<task>``) evaluates the same checkpoint on
    every object and scores the WORST object, so one strong object cannot
    carry the grade. ``challenges`` are extra scenarios evaluated in addition
    to the default and reported alongside it; ``challenge`` makes this call
    evaluate one scenario instead of the default. ``allow_privileged`` lets
    an adaptation-track actor read the sampled physics, to measure a
    privileged reference policy (never used for grading).
    """
    parts_specs = spec.per_object()
    if len(parts_specs) > 1:
        started = time.perf_counter()
        parts = [
            evaluate_cell(
                part,
                checkpoint,
                config=config,
                episodes=episodes,
                num_envs=num_envs,
                seed=seed,
                device=device,
                rubric=rubric,
                root_dir=root_dir,
                log=log,
                challenge=challenge,
                challenges=challenges,
                allow_privileged=allow_privileged,
            )
            for part in parts_specs
        ]
        errors = [p.error for p in parts if p.error]
        worst = min(parts, key=lambda p: p.score.value if p.score.value is not None else -1.0)
        result = CellResult(
            cell=spec.key,
            checkpoint=worst.checkpoint,
            config=str(config) if config is not None else None,
            metrics=None if errors else worst.metrics,
            score=score_cell(None if errors else worst.metrics, spec, rubric),
            warnings=[w for p in parts for w in p.warnings],
            error="; ".join(errors) if errors else None,
            parts=parts,
        )
        result.seconds = time.perf_counter() - started
        return result

    started = time.perf_counter()
    warnings: list[str] = []
    ckpt_text = str(checkpoint)
    label = spec.key if challenge is None else f"{spec.key} @{challenge.name}"
    if challenge is not None and challenge.episodes is not None:
        episodes = int(challenge.episodes)
    try:
        ckpt_path = resolve_checkpoint(checkpoint)
        ckpt_text = str(ckpt_path)
        cfg, drift = compose_evaluation(spec, config, allow_privileged=allow_privileged)
        if challenge is not None:
            cfg = challenge.apply(cfg)
        if drift:
            warnings.append(
                "evaluator-owned fields in the submitted config were ignored: " + ", ".join(drift)
            )
        rounds = max(1, int(math.ceil(int(episodes) / int(num_envs))))
        env = create_env(cfg, num_envs=int(num_envs), root_dir=root_dir)
        try:
            loaded = load_policy(cfg, env, ckpt_path, device=device)
            log(
                f"[{label}] {ckpt_path.name}: policy_obs_mode={loaded.policy_obs_mode}, "
                f"{rounds} x {num_envs} episodes"
            )
            metrics = run_evaluation_episodes(
                env, loaded.wrapped_env, loaded.policy, rounds=rounds, seed=seed
            )
        finally:
            env.close()
        result = CellResult(
            cell=label,
            checkpoint=ckpt_text,
            config=str(config) if config is not None else None,
            metrics=metrics,
            score=score_cell(metrics, spec, rubric),
            warnings=warnings,
        )
    except (CheckpointError, StudentConfigError, FileNotFoundError, ValueError) as exc:
        result = CellResult(
            cell=label,
            checkpoint=ckpt_text,
            config=str(config) if config is not None else None,
            metrics=None,
            score=score_cell(None, spec, rubric),
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}",
        )
    if challenge is None and challenges:
        result.challenges = [
            evaluate_cell(
                spec,
                checkpoint,
                config=config,
                episodes=episodes,
                num_envs=num_envs,
                seed=seed,
                device=device,
                rubric=rubric,
                root_dir=root_dir,
                log=log,
                challenge=item,
                allow_privileged=allow_privileged,
            )
            for item in challenges
        ]
    result.seconds = time.perf_counter() - started
    return result


def load_manifest(manifest: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(manifest, dict):
        return dict(manifest), None
    path = Path(manifest)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must contain a mapping")
    return data, path.parent


def _resolve(base: Path | None, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def evaluate_submission(
    manifest: str | Path | dict[str, Any],
    *,
    episodes: int | None = None,
    num_envs: int = 16,
    seed: int = 0,
    device: str = "cpu",
    rubric: Rubric | None = None,
    cells: list[str] | None = None,
    root_dir: str | Path = ".",
    log=print,
    challenges: Sequence[Challenge] = (),
    allow_privileged: bool = False,
) -> dict[str, Any]:
    rubric = rubric or Rubric.load()
    data, base = load_manifest(manifest)
    team = str(data.get("team", "unknown"))
    default_config = data.get("config")
    default_config_path = _resolve(base, str(default_config)) if default_config else None
    declared = data.get("cells") or {}
    if not isinstance(declared, dict):
        raise ValueError("manifest 'cells' must be a mapping of cell -> checkpoint")
    wanted = set(cells) if cells else None
    n_episodes = int(episodes) if episodes is not None else int(rubric.default_episodes)

    results: list[CellResult] = []
    for cell, entry in declared.items():
        if wanted is not None and cell not in wanted:
            continue
        try:
            spec = parse_cell(str(cell))
        except ValueError as exc:
            results.append(
                CellResult(
                    cell=str(cell),
                    checkpoint=None,
                    config=None,
                    metrics=None,
                    score=CellScore(rubric.metric, None, 0, 0.0, rubric.max_points, []),
                    error=f"ValueError: {exc}",
                )
            )
            continue
        if isinstance(entry, dict):
            checkpoint = _resolve(base, str(entry.get("checkpoint", "")))
            config_entry = entry.get("config")
            config_path = _resolve(base, str(config_entry)) if config_entry else default_config_path
        else:
            checkpoint = _resolve(base, str(entry))
            config_path = default_config_path
        results.append(
            evaluate_cell(
                spec,
                checkpoint,
                config=config_path,
                episodes=n_episodes,
                num_envs=num_envs,
                seed=seed,
                device=device,
                rubric=rubric,
                root_dir=root_dir,
                log=log,
                challenges=challenges,
                allow_privileged=allow_privileged,
            )
        )
    total = sum(r.score.points for r in results)
    max_total = rubric.max_points * len(results)
    return {
        "team": team,
        "members": list(data.get("members", [])),
        "episodes_per_cell": n_episodes,
        "seed": seed,
        "total_points": total,
        "max_points": max_total,
        "cells": [r.to_dict() for r in results],
    }


def write_json(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def format_cell_line(result: CellResult) -> str:
    value = "-" if result.score.value is None else f"{result.score.value:.2f}"
    status = f"ERROR {result.error}" if result.error else f"tier {result.score.tier}"
    line = (
        f"{result.cell:32s} {value:>6s} goals/ep  {result.score.points:>4.1f}/"
        f"{result.score.max_points:.0f} pts  {status}"
    )
    for item in result.challenges:
        value = "-" if item.score.value is None else f"{item.score.value:.2f}"
        drop = "-" if item.metrics is None else f"{item.metrics.drop_rate:.2f}"
        tail = f"ERROR {item.error}" if item.error else f"drop {drop}"
        line += f"\n  {item.cell:30s} {value:>6s} goals/ep  {tail}"
    return line


__all__ = [
    "CellResult",
    "evaluate_cell",
    "evaluate_submission",
    "format_cell_line",
    "load_manifest",
    "write_json",
]
