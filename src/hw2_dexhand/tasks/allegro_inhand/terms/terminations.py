"""Termination terms for the Allegro in-hand tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.observations import _ObjectRootTerm

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

_OBJECT = SceneEntityCfg("object")


class object_dropped(_ObjectRootTerm):
    """True when the object's root body falls below ``z_threshold`` (world z).

    The mixed pool passes ``z_thresholds`` (one per object) instead.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        thresholds = cfg.params.get("z_thresholds")
        if thresholds is not None:
            # Mixed pool: one threshold per object, in tensor_body_names order,
            # so every env terminates exactly as its object's single scene does.
            values = [float(v) for v in thresholds]
            if len(values) != self._object.num_objects:
                raise ValueError(
                    f"{self.name} z_thresholds has {len(values)} entries for "
                    f"{self._object.num_objects} objects"
                )
            table = torch.tensor(values, dtype=torch.float32, device=self._device)
            self._z_threshold = table[self._object.active_ids]
        else:
            self._z_threshold = torch.full(
                (env.num_envs,),
                float(cfg.params["z_threshold"]),
                dtype=torch.float32,
                device=self._device,
            )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        pos = self._value(env, "pos_w")
        if pos is None:
            return torch.zeros((env.num_envs,), dtype=torch.bool, device=self._device)
        return pos[:, 2] < self._z_threshold


class goal_count_reached(ManagerTermBase):
    """True once the goal command's per-episode success count reaches its limit.

    Params: ``command_name`` (default ``goal``). Never fires when the command's
    ``max_consecutive_successes`` is 0.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._command_name = str(cfg.params.get("command_name", "goal"))
        self._device = torch.device(getattr(env, "device", torch.device("cpu")))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        term = env.command_manager.get_term(self._command_name)
        limit = int(getattr(term, "max_consecutive_successes", 0))
        if limit <= 0:
            return torch.zeros((env.num_envs,), dtype=torch.bool, device=self._device)
        return term.goals_reached >= float(limit)
