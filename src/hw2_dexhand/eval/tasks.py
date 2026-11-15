"""The HW2 task matrix: 4 goal tasks x 3 objects = 12 reorientation cells.

A cell is ``<task>/<object>``, e.g. ``yaw_absolute/ball``. Each cell maps to a
packaged owner (``allegro_reorient/<object>``) plus the two goal overrides; the
evaluator and the training command compose exactly the same thing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

TASKS: tuple[str, ...] = ("yaw_absolute", "yaw_relative", "so3_absolute", "so3_relative")
OBJECTS: tuple[str, ...] = ("ball", "cube", "cylinder")

_TASK_GOAL: dict[str, tuple[str, str]] = {
    "yaw_absolute": ("yaw", "absolute"),
    "yaw_relative": ("yaw", "relative"),
    "so3_absolute": ("so3", "absolute"),
    "so3_relative": ("so3", "relative"),
}
_TASK_TITLE: dict[str, str] = {
    "yaw_absolute": "yaw goal, world frame",
    "yaw_relative": "yaw goal, relative to the reset pose",
    "so3_absolute": "SO(3) goal, world frame",
    "so3_relative": "SO(3) goal, relative to the reset pose",
}
TASK_GROUP = "allegro_reorient"
ENV_NAME = "AllegroInhandReorient"
TRACKS: tuple[str, ...] = ("single", "multi", "adapt")
MULTI_OWNER = "allegro_multi/train"
ADAPT_GROUP = "allegro_adapt"
# Multi-object contract: the policy never observes which object it holds, so the
# single-object evaluation scenes drop the object-orientation term.
MULTI_EVAL_OVERRIDES: tuple[str, ...] = ("env.observations.policy.terms.object_quat=null",)
# Adaptation contract: graded under a HELD-OUT physics range, wider than training.
ADAPT_EVAL_OVERRIDES: tuple[str, ...] = (
    "env.events.randomize_object_physics.params.mass_scale_range=[0.3,2.0]",
    "env.events.randomize_object_physics.params.friction_scale_range=[0.4,1.8]",
    "env.events.randomize_object_physics.params.com_offset_range=[-0.02,0.02]",
    "env.events.randomize_hand_gains.params.kp_scale_range=[0.5,1.6]",
)
# Adaptation contract: at grading the two sampled-value terms are removed from
# the environment, so no actor class can read them through the observation
# tensor. ``--allow-privileged`` keeps them, to measure a privileged reference
# policy; it is never used for grading.
ADAPT_BLINDFOLD_OVERRIDES: tuple[str, ...] = (
    "env.observations.critic.terms.object_physics=null",
    "env.observations.critic.terms.hand_gains=null",
)


@dataclass(frozen=True)
class TaskSpec:
    """One cell of the matrix and how it is expressed in HW2 configs.

    ``track``: ``single`` (one object, ``<task>/<object>``), ``multi`` (one
    policy for all objects, ``multi/<task>``; ``obj`` is the evaluation object
    or ``None`` for the training owner) or ``adapt`` (randomized physics,
    ``adapt/<task>/<object>``).
    """

    task: str
    obj: str | None
    track: str = "single"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.track not in TRACKS:
            raise ValueError(f"track must be one of {TRACKS}, got {self.track!r}")
        if self.obj is None and self.track != "multi":
            raise ValueError(f"{self.track} cells need an object")
        if self.obj is not None and self.obj not in OBJECTS:
            raise ValueError(f"object must be one of {OBJECTS}, got {self.obj!r}")

    @property
    def key(self) -> str:
        if self.track == "single":
            return f"{self.task}/{self.obj}"
        if self.track == "multi":
            return f"multi/{self.task}" if self.obj is None else f"multi/{self.task}/{self.obj}"
        return f"adapt/{self.task}/{self.obj}"

    @property
    def title(self) -> str:
        if self.track == "multi" and self.obj is None:
            return f"all objects, one policy / {_TASK_TITLE[self.task]}"
        prefix = {"single": "", "multi": "multi-object policy, ", "adapt": "held-out physics, "}
        return f"{prefix[self.track]}{self.obj} / {_TASK_TITLE[self.task]}"

    def per_object(self) -> list[TaskSpec]:
        """A multi training cell expands to one evaluation cell per object."""
        if self.track != "multi" or self.obj is not None:
            return [self]
        return [TaskSpec(self.task, obj, "multi") for obj in OBJECTS]

    @property
    def goal_mode(self) -> str:
        return _TASK_GOAL[self.task][0]

    @property
    def goal_frame(self) -> str:
        return _TASK_GOAL[self.task][1]

    @property
    def owner(self) -> str:
        """Hydra ``task=`` choice of the packaged reference owner.

        Training owner for a multi cell without object; evaluation owner (the
        single-object scene) for a multi cell with an object.
        """
        if self.track == "multi":
            return MULTI_OWNER if self.obj is None else f"{TASK_GROUP}/{self.obj}"
        if self.track == "adapt":
            return f"{ADAPT_GROUP}/{self.obj}"
        return f"{TASK_GROUP}/{self.obj}"

    def goal_overrides(self) -> list[str]:
        return [
            f"env.commands.goal.mode={self.goal_mode}",
            f"env.commands.goal.frame={self.goal_frame}",
        ]

    def eval_overrides(self) -> list[str]:
        """Evaluator-owned overrides that define the track's grading contract."""
        if self.track == "multi" and self.obj is not None:
            return list(MULTI_EVAL_OVERRIDES)
        if self.track == "adapt":
            return list(ADAPT_EVAL_OVERRIDES)
        return []

    def hydra_overrides(self) -> list[str]:
        return [f"task={self.owner}", *self.goal_overrides()]

    def train_command(self, config: str | None = None) -> str:
        parts = ["hw2 train", self.key]
        if config is not None:
            parts += ["--config", config]
        return " ".join(parts)


TASK_MATRIX: tuple[TaskSpec, ...] = (
    *(TaskSpec(task, obj) for task in TASKS for obj in OBJECTS),
    *(TaskSpec(task, None, "multi") for task in TASKS),
    *(TaskSpec(task, obj, "adapt") for task in TASKS for obj in OBJECTS),
)
_BY_KEY: dict[str, TaskSpec] = {spec.key: spec for spec in TASK_MATRIX}
_BY_KEY.update({spec.key: spec for multi in TASK_MATRIX for spec in multi.per_object()})


def parse_cell(text: str) -> TaskSpec:
    """Parse a cell key; a packaged owner (``allegro_reorient/ball``,
    ``allegro_multi/train``, ``allegro_adapt/cube``) means its default goal."""
    key = text.strip().strip("/")
    if key in _BY_KEY:
        return _BY_KEY[key]
    parts = key.split("/")
    if len(parts) == 2 and parts[0] == TASK_GROUP and parts[1] in OBJECTS:
        return _BY_KEY[f"yaw_absolute/{parts[1]}"]
    if key == MULTI_OWNER:
        return _BY_KEY["multi/yaw_absolute"]
    if len(parts) == 2 and parts[0] == ADAPT_GROUP and parts[1] in OBJECTS:
        return _BY_KEY[f"adapt/yaw_absolute/{parts[1]}"]
    raise ValueError(
        f"unknown cell {text!r}; expected <task>/<object>, multi/<task> or "
        f"adapt/<task>/<object> with task in {TASKS} and object in {OBJECTS}"
    )


def iter_cells(
    task: str | None = None, obj: str | None = None, track: str | None = None
) -> Iterable[TaskSpec]:
    for spec in TASK_MATRIX:
        if task is not None and spec.task != task:
            continue
        if obj is not None and spec.obj != obj:
            continue
        if track is not None and spec.track != track:
            continue
        yield spec


__all__ = [
    "ADAPT_BLINDFOLD_OVERRIDES",
    "ADAPT_EVAL_OVERRIDES",
    "ADAPT_GROUP",
    "ENV_NAME",
    "MULTI_EVAL_OVERRIDES",
    "MULTI_OWNER",
    "TRACKS",
    "OBJECTS",
    "TASKS",
    "TASK_GROUP",
    "TASK_MATRIX",
    "TaskSpec",
    "iter_cells",
    "parse_cell",
]
