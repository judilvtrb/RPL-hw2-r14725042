"""One way to read "the object" for single-object and mixed-object scenes.

Single-object owners declare an entity whose root body is the object
(``asset_cfg: {name: object}``). The mixed pool declares a read-only entity
packing all object bodies (``entity_name: objects``,
``tensor_body_names: [object_ball, object_cube, object_cylinder]``) and the
env's active object is ``env_id % num_objects``. :class:`ObjectView` hides
the difference: every object term asks it for a ``(num_envs, k)`` field.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


def device_of(env: ManagerBasedRlEnv) -> torch.device:
    return torch.device(getattr(env, "device", torch.device("cpu")))


def ready_read_plan(env: ManagerBasedRlEnv) -> Any | None:
    read_plan = getattr(env.scene, "read_plan", None)
    if read_plan is None or not getattr(read_plan, "ready", True):
        return None
    return read_plan


def resolve_root_body(env: ManagerBasedRlEnv, entity_name: str, term_name: str) -> str:
    scene_cfg = env.cfg.scene
    entity_cfg = scene_cfg.entities.get(entity_name) if scene_cfg is not None else None
    root = getattr(entity_cfg, "root_body_name", None) if entity_cfg is not None else None
    if not isinstance(root, str) or not root:
        raise ValueError(f"{term_name} requires entity '{entity_name}' to declare root_body_name")
    return root


def active_object_ids(env: ManagerBasedRlEnv, num_objects: int) -> torch.Tensor:
    """Per-env index of the active object in a mixed pool: ``env_id % num_objects``."""
    if num_objects <= 0:
        raise ValueError("active_object_ids requires at least one object")
    return torch.arange(env.num_envs, device=device_of(env), dtype=torch.int64) % num_objects


class ObjectView:
    """Cold-path binding to the object body (or bodies) a term reads.

    ``params`` are the term's YAML params: either ``asset_cfg`` (single root
    entity) or ``entity_name`` + ``tensor_body_names`` (mixed pool).
    """

    def __init__(self, env: ManagerBasedRlEnv, params: dict[str, Any], term_name: str):
        self.device = device_of(env)
        body_names = params.get("tensor_body_names")
        if body_names is not None:
            entity_name = params.get("entity_name")
            if not isinstance(entity_name, str) or not entity_name:
                raise TypeError(f"{term_name} with tensor_body_names requires entity_name")
            if (
                not isinstance(body_names, Sequence)
                or isinstance(body_names, str)
                or not body_names
            ):
                raise TypeError(f"{term_name} tensor_body_names must be a non-empty list")
            self.entity_name = entity_name
            self.body_names = tuple(str(name) for name in body_names)
            self.multi = True
        else:
            asset_cfg = params.get("asset_cfg", SceneEntityCfg("object"))
            if not isinstance(asset_cfg, SceneEntityCfg):
                raise TypeError(f"{term_name} asset_cfg must be a SceneEntityCfg")
            self.entity_name = asset_cfg.name
            self.body_names = (resolve_root_body(env, self.entity_name, term_name),)
            self.multi = False
        self.rows = torch.arange(env.num_envs, device=self.device, dtype=torch.int64)
        self.active_ids = (
            active_object_ids(env, len(self.body_names))
            if self.multi
            else torch.zeros((env.num_envs,), dtype=torch.int64, device=self.device)
        )

    @property
    def num_objects(self) -> int:
        return len(self.body_names)

    def view(self, env: ManagerBasedRlEnv) -> Any | None:
        read_plan = ready_read_plan(env)
        if read_plan is None:
            return None
        return read_plan.body_tensor_view(self.entity_name, self.body_names)

    def field(self, env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
        """``(num_envs, k)`` of ``pos_w`` / ``quat_w`` / ``lin_vel_w`` / ``ang_vel_w``,
        or ``None`` before the read plan exists (the managers' sizing probe)."""
        view = self.view(env)
        if view is None:
            return None
        values = getattr(view, name)
        if self.multi:
            return values[self.rows, self.active_ids, :]
        return values[:, 0, :]


__all__ = ["ObjectView", "active_object_ids", "device_of", "ready_read_plan", "resolve_root_body"]
