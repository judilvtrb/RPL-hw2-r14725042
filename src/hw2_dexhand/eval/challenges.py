"""Challenge sets: extra evaluator-owned scenarios applied on top of a cell.

``hw2 eval|grade ... --challenge public`` runs the packaged public set
(``public_challenges.yaml``, the adaptation track's self-check); the hidden
grading sets are files kept outside the repository. A challenge file is YAML::

    challenges:
      - name: heavy_slippery_weak
        description: corner of the held-out range
        overrides:                       # Hydra-style dotlist, applied last
          - env.events.randomize_object_physics.params.mass_scale_range=[1.9,2.0]
          - env.events.randomize_object_physics.params.friction_scale_range=[0.4,0.45]
          - env.events.randomize_hand_gains.params.kp_scale_range=[0.5,0.55]
      - name: strong_motor_pinned
        config:                          # nested mapping, merged last
          env:
            events:
              randomize_hand_gains:
                params: {kp_scale_range: [1.6, 1.6]}
        episodes: 16

Scenarios run on top of the cell's evaluation config (student overlay with
evaluator-owned fields re-imposed, privileged terms dropped), so a student
config cannot neutralise them. Results are reported per scenario next to the
cell's default score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from hw2_dexhand import PACKAGE_ROOT

# Names ``--challenge`` accepts in place of a file path.
PACKAGED_CHALLENGES: dict[str, Path] = {"public": PACKAGE_ROOT / "eval" / "public_challenges.yaml"}


@dataclass(frozen=True)
class Challenge:
    name: str
    description: str = ""
    overrides: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    episodes: int | None = None

    def apply(self, cfg: DictConfig) -> DictConfig:
        """Return ``cfg`` with this scenario merged on top."""
        OmegaConf.set_struct(cfg, False)
        if self.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(self.overrides)))
        if self.config:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(self.config))
        assert isinstance(cfg, DictConfig)
        OmegaConf.resolve(cfg)
        return cfg


def resolve_challenge_path(name_or_path: str | Path) -> Path:
    """A packaged set name (``public``) or a path to a challenge file."""
    key = str(name_or_path)
    if key in PACKAGED_CHALLENGES:
        return PACKAGED_CHALLENGES[key]
    return Path(name_or_path)


def load_challenges(path: str | Path) -> list[Challenge]:
    path = resolve_challenge_path(path)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("challenges") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty 'challenges' list")
    challenges: list[Challenge] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"{path}: every challenge needs a name")
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"{path}: duplicate challenge name {name!r}")
        seen.add(name)
        overrides = entry.get("overrides") or []
        if not isinstance(overrides, Sequence) or isinstance(overrides, str):
            raise ValueError(f"{path}: challenge {name!r} overrides must be a list")
        config = entry.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError(f"{path}: challenge {name!r} config must be a mapping")
        episodes = entry.get("episodes")
        challenges.append(
            Challenge(
                name=name,
                description=str(entry.get("description", "")),
                overrides=tuple(str(o) for o in overrides),
                config=dict(config),
                episodes=int(episodes) if episodes is not None else None,
            )
        )
    return challenges


__all__ = ["PACKAGED_CHALLENGES", "Challenge", "load_challenges", "resolve_challenge_path"]
