"""Mixed-object pool: several objects in one scene, one active per env.

UniLab's ``fixed_model_variants`` only accept same-topology variants (identical
geom types), so ball, cube and cylinder cannot be variants of one model. The
mixed pool instead keeps all objects in every env: at reset the env's *active*
object is placed on the palm and the others are parked on the floor far from
the hand. The policy only ever observes the active object, so its I/O is the
same as in the single-object tasks.

Entities (declared in the owner YAML):

* one root entity per object (``object_ball`` ...) that owns the reset writes;
* one read-only entity (``objects``) listing all object bodies, which the
  observation and termination terms read through the packed plan and gather
  with the per-env active index.

The active assignment is deterministic, ``env_id % num_objects``, which
mirrors the variant plan's round-robin and keeps the terms stateless.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

from hw2_dexhand.tasks.allegro_inhand.terms.observations import _device

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


from hw2_dexhand.tasks.allegro_inhand.terms import observations as _observations  # noqa: E402
from hw2_dexhand.tasks.allegro_inhand.terms import terminations as _terminations  # noqa: E402
from hw2_dexhand.tasks.allegro_inhand.terms.object_view import active_object_ids  # noqa: E402

# The generic object terms read the env's active object when given
# ``entity_name`` + ``tensor_body_names``; these names are kept for owners
# written against the mixed-pool names.
active_object_pos_w = _observations.object_pos_w
active_object_quat_w = _observations.object_quat_w
active_object_dropped = _terminations.object_dropped


def _pose_table(
    positions: Mapping[str, Sequence[float]],
    entities: Sequence[str],
    *,
    label: str,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Build ``(num_objects, num_envs, 13)`` root states: pos, identity quat, zero vel."""
    table = torch.zeros((len(entities), num_envs, 13), dtype=torch.float32, device=device)
    table[:, :, 3] = 1.0
    for index, name in enumerate(entities):
        if name not in positions:
            raise KeyError(f"place_active_object {label} is missing entity '{name}'")
        xyz = [float(v) for v in positions[name]]
        if len(xyz) != 3:
            raise ValueError(f"place_active_object {label}['{name}'] must have 3 numbers")
        table[index, :, 0:3] = torch.tensor(xyz, dtype=torch.float32, device=device)
    return table


class place_active_object(ManagerTermBase):
    """Reset event: active object to its palm pose, the others to park poses.

    Params: ``object_entities`` (root entities in active-index order),
    ``active_positions`` and ``park_positions`` (entity name -> ``[x, y, z]``).
    Runs after ``reset_scene_to_default`` so the hand is already at the keyframe.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        entities = cfg.params.get("object_entities")
        if not isinstance(entities, Sequence) or isinstance(entities, str) or not entities:
            raise TypeError("place_active_object requires a non-empty object_entities list")
        self._entities = tuple(str(name) for name in entities)
        for name in self._entities:
            env.scene[name]  # fail at construction if an entity is missing
        self._device = _device(env)
        self._ids = active_object_ids(env, len(self._entities))
        self._active = _pose_table(
            cfg.params["active_positions"],
            self._entities,
            label="active_positions",
            num_envs=env.num_envs,
            device=self._device,
        )
        self._park = _pose_table(
            cfg.params["park_positions"],
            self._entities,
            label="park_positions",
            num_envs=env.num_envs,
            device=self._device,
        )
        origins = env.scene.env_origins(self._device).to(dtype=torch.float32)
        self._active[:, :, 0:3] += origins
        self._park[:, :, 0:3] += origins

    @property
    def active_ids(self) -> torch.Tensor:
        return self._ids

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **params: Any) -> None:
        del params
        reset_mask = env.event_term_mask_buffer.scatter(env_ids)
        for index, name in enumerate(self._entities):
            is_active = (self._ids == index).unsqueeze(1)
            root_states = torch.where(is_active, self._active[index], self._park[index])
            env.scene[name].write_root_state_tensor_to_sim(root_states, reset_mask)


class reset_from_grasp_caches(ManagerTermBase):
    """Mixed-pool reset: the active object and the hand from that object's cache;
    the other objects parked.

    Params: ``object_entities`` (root entities in active-index order),
    ``cache_paths`` (entity name -> grasp cache file, same (N, 23) layout as the
    single-object caches), ``park_positions`` (entity name -> ``[x, y, z]``),
    ``hand_asset_cfg`` (default entity ``robot``), ``action_name`` (default
    ``hand``), ``joint_noise`` (default 0). Exposes ``init_hand_pose``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        from unilab.managers.scene_entity_config import SceneEntityCfg

        from hw2_dexhand.tasks.allegro_inhand.terms.events import (
            NUM_HAND_JOINTS,
            load_grasp_cache,
        )

        entities = cfg.params.get("object_entities")
        if not isinstance(entities, Sequence) or isinstance(entities, str) or not entities:
            raise TypeError(f"{self.name} requires a non-empty object_entities list")
        self._entities = tuple(str(name) for name in entities)
        self._device = _device(env)
        cache_paths = cfg.params.get("cache_paths")
        if not isinstance(cache_paths, Mapping):
            raise TypeError(f"{self.name} requires cache_paths (entity name -> file)")
        self._caches = []
        for name in self._entities:
            env.scene[name]
            if name not in cache_paths:
                raise KeyError(f"{self.name} cache_paths is missing entity '{name}'")
            self._caches.append(
                torch.from_numpy(load_grasp_cache(str(cache_paths[name]))).to(self._device)
            )
        hand_cfg = cfg.params.get("hand_asset_cfg", SceneEntityCfg("robot"))
        if not isinstance(hand_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} hand_asset_cfg must be a SceneEntityCfg")
        self._hand_name = hand_cfg.name
        self._hand_joint_ids = hand_cfg.joint_ids
        self._action_name = str(cfg.params.get("action_name", "hand"))
        self._joint_noise = float(cfg.params.get("joint_noise", 0.0))
        self._num_hand_joints = NUM_HAND_JOINTS
        self._ids = active_object_ids(env, len(self._entities))
        self._park = _pose_table(
            cfg.params["park_positions"],
            self._entities,
            label="park_positions",
            num_envs=env.num_envs,
            device=self._device,
        )
        self._origins = env.scene.env_origins(self._device).to(dtype=torch.float32)
        self._park[:, :, 0:3] += self._origins
        self._init_hand_pose = torch.zeros(
            (env.num_envs, NUM_HAND_JOINTS), dtype=torch.float32, device=self._device
        )
        self._zero_vel = torch.zeros_like(self._init_hand_pose)

    @property
    def init_hand_pose(self) -> torch.Tensor:
        return self._init_hand_pose

    @property
    def active_ids(self) -> torch.Tensor:
        return self._ids

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any) -> None:
        del params
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device)
        reset_mask = env.event_term_mask_buffer.scatter(env_ids)
        hand = torch.zeros_like(self._init_hand_pose)
        for index, name in enumerate(self._entities):
            cache = self._caches[index]
            rows = env.rng.integers(0, int(cache.shape[0]), (env.num_envs,))
            sample = cache[rows]
            is_active = self._ids == index
            root = self._park[index].clone()
            root[is_active, 0:3] = sample[is_active, 16:19] + self._origins[is_active]
            root[is_active, 3:7] = sample[is_active, 19:23]
            root[is_active, 7:13] = 0.0
            env.scene[name].write_root_state_tensor_to_sim(root, reset_mask)
            hand = torch.where(is_active.unsqueeze(1), sample[:, : self._num_hand_joints], hand)
        if self._joint_noise > 0.0:
            hand = hand + env.rng.uniform(
                -self._joint_noise, self._joint_noise, tuple(hand.shape), dtype=hand.dtype
            )
        env.scene[self._hand_name].write_joint_state_to_sim_mask(
            hand, self._zero_vel, reset_mask, joint_ids=self._hand_joint_ids
        )
        set_target = getattr(env.action_manager.get_term(self._action_name), "set_target", None)
        if not callable(set_target):
            raise TypeError(
                f"{self.name} requires action term '{self._action_name}' with set_target"
            )
        set_target(hand[env_ids], env_ids)
        self._init_hand_pose = torch.where(reset_mask.unsqueeze(1), hand, self._init_hand_pose)


__all__ = [
    "active_object_dropped",
    "active_object_ids",
    "active_object_pos_w",
    "active_object_quat_w",
    "place_active_object",
    "reset_from_grasp_caches",
]
