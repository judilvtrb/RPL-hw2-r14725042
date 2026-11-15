"""Rubric: map a cell's evaluation metric to tier points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hw2_dexhand import PACKAGE_ROOT
from hw2_dexhand.eval.rollout import EvalMetrics
from hw2_dexhand.eval.tasks import OBJECTS, TASKS, TaskSpec

DEFAULT_RUBRIC_PATH = PACKAGE_ROOT / "eval" / "rubric.yaml"


@dataclass
class Rubric:
    metric: str = "goals_per_episode"
    points: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    tiers: dict[str, list[float]] = field(default_factory=dict)
    # Adaptation cells are graded under held-out physics, a different scale from
    # the single-object cells; tasks listed here use these thresholds instead.
    adapt_tiers: dict[str, list[float]] = field(default_factory=dict)
    # Multi-object cells score the WORST of three objects with one policy that
    # never sees which object it holds, a harder scale again; same rule.
    multi_tiers: dict[str, list[float]] = field(default_factory=dict)
    default_episodes: int = 64

    def _check_thresholds(self, label: str, thresholds: list[float]) -> None:
        if len(thresholds) != len(self.points):
            raise ValueError(
                f"rubric.{label} has {len(thresholds)} thresholds but "
                f"rubric.points has {len(self.points)} entries"
            )
        if any(b < a for a, b in zip(thresholds, thresholds[1:], strict=False)):
            raise ValueError(f"rubric.{label} must be ascending, got {thresholds}")

    @staticmethod
    def _check_key(label: str, key: str) -> None:
        """A table key is ``<task>`` or ``<task>/<object>``."""
        task, _, obj = key.partition("/")
        if task not in TASKS or (obj and obj not in OBJECTS):
            raise ValueError(f"rubric.{label} names unknown task {key!r}")

    def validate(self) -> None:
        if not self.points or any(p < 0 for p in self.points):
            raise ValueError("rubric.points must be a non-empty list of non-negative numbers")
        for task in TASKS:
            if task not in self.tiers:
                raise ValueError(f"rubric.tiers is missing task {task!r}")
        for label, table in (
            ("tiers", self.tiers),
            ("adapt_tiers", self.adapt_tiers),
            ("multi_tiers", self.multi_tiers),
        ):
            for key, thresholds in table.items():
                self._check_key(label, key)
                self._check_thresholds(f"{label}[{key!r}]", thresholds)
        if int(self.default_episodes) <= 0:
            raise ValueError("rubric.default_episodes must be positive")

    def thresholds_for(self, spec: TaskSpec) -> list[float]:
        """Tier thresholds for a cell.

        The table is ``adapt_tiers`` on the adaptation track and ``multi_tiers`` on
        the multi-object track (the pooled cell and its per-object parts alike),
        falling back to ``tiers`` when the task has no entry there. Within a table
        a ``<task>/<object>`` key wins over the task-wide ``<task>`` key, so objects
        of different difficulty can carry their own thresholds.
        """
        tables = [{"adapt": self.adapt_tiers, "multi": self.multi_tiers}.get(spec.track, {})]
        tables.append(self.tiers)
        keys = ([f"{spec.task}/{spec.obj}"] if spec.obj else []) + [spec.task]
        for table in tables:
            for key in keys:
                if key in table:
                    return list(table[key])
        raise KeyError(f"rubric has no thresholds for {spec.key}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        def table(key: str) -> dict[str, list[float]]:
            return {str(k): [float(x) for x in v] for k, v in dict(data.get(key) or {}).items()}

        rubric = cls(
            metric=str(data.get("metric", "goals_per_episode")),
            points=[float(x) for x in data.get("points", [1.0, 2.0, 3.0])],
            tiers=table("tiers"),
            adapt_tiers=table("adapt_tiers"),
            multi_tiers=table("multi_tiers"),
            default_episodes=int(data.get("default_episodes", 64)),
        )
        rubric.validate()
        return rubric

    @classmethod
    def load(cls, path: str | Path | None = None) -> Rubric:
        rubric_path = Path(path) if path is not None else DEFAULT_RUBRIC_PATH
        data = yaml.safe_load(rubric_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"rubric file {rubric_path} must contain a mapping")
        return cls.from_dict(data)

    @property
    def max_points(self) -> float:
        return float(max(self.points))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellScore:
    metric: str
    value: float | None
    tier: int  # 0 = no tier reached, 1..len(points)
    points: float
    max_points: float
    thresholds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_cell(metrics: EvalMetrics | None, spec: TaskSpec, rubric: Rubric) -> CellScore:
    """Map a cell's metrics to rubric points; ``metrics=None`` scores zero."""
    thresholds = rubric.thresholds_for(spec)
    value: float | None = None
    if metrics is not None and metrics.num_episodes > 0:
        value = float(metrics.summary()[rubric.metric])
    tier = 0
    if value is not None and value == value:  # not NaN
        for index, threshold in enumerate(thresholds, start=1):
            if value >= threshold:
                tier = index
    points = float(rubric.points[tier - 1]) if tier > 0 else 0.0
    return CellScore(
        metric=rubric.metric,
        value=value,
        tier=tier,
        points=points,
        max_points=rubric.max_points,
        thresholds=list(thresholds),
    )


__all__ = ["DEFAULT_RUBRIC_PATH", "CellScore", "Rubric", "score_cell"]
