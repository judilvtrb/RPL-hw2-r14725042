"""Reset events for the Allegro in-hand tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.observations import _device

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

NUM_HAND_JOINTS = 16
CACHE_WIDTH = NUM_HAND_JOINTS + 3 + 4  # hand qpos, object position, object quat (w, x, y, z)


def load_grasp_cache(path: str | Path) -> np.ndarray:
    """Load an ``(N, 23)`` grasp table: 16 hand joints, object xyz, object wxyz quat."""
    cache_file = Path(path)
    if not cache_file.is_file():
        raise FileNotFoundError(
            f"Grasp cache not found: {cache_file}. Generate it with `hw2 make-caches` "
            "or point the owner's reset_from_grasp_cache.params.cache_path at an existing file."
        )
    table = np.load(cache_file)
    if table.ndim != 2 or table.shape[1] != CACHE_WIDTH or table.shape[0] == 0:
        raise ValueError(
            f"Grasp cache {cache_file} must have shape (N>0, {CACHE_WIDTH}), got {table.shape}"
        )
    if not np.isfinite(table).all():
        raise ValueError(f"Grasp cache {cache_file} contains non-finite values")
    return np.asarray(table, dtype=np.float32)


class reset_from_grasp_cache(ManagerTermBase):
    """Reset the hand joints and the object pose from a cached stable grasp.

    Params: ``cache_path``; ``hand_asset_cfg`` (default entity ``robot``);
    ``object_asset_cfg`` (default entity ``object``); ``action_name`` (the
    integrated-target action to seed, default ``hand``); ``joint_noise`` (uniform
    ±rad added to the cached hand pose, default 0). Runs after
    ``reset_scene_to_default`` and overrides the keyframe for the reset rows.
    The hand pose written at the last reset is exposed as ``init_hand_pose``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._device = _device(env)
        self._cache = torch.from_numpy(load_grasp_cache(str(cfg.params["cache_path"]))).to(
            self._device
        )
        hand_cfg = cfg.params.get("hand_asset_cfg", SceneEntityCfg("robot"))
        object_cfg = cfg.params.get("object_asset_cfg", SceneEntityCfg("object"))
        if not isinstance(hand_cfg, SceneEntityCfg) or not isinstance(object_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} hand_asset_cfg/object_asset_cfg must be SceneEntityCfg")
        self._hand_name = hand_cfg.name
        self._object_name = object_cfg.name
        self._hand_joint_ids = hand_cfg.joint_ids
        self._action_name = str(cfg.params.get("action_name", "hand"))
        self._joint_noise = float(cfg.params.get("joint_noise", 0.0))
        for name in (self._hand_name, self._object_name):
            env.scene[name]  # fail at construction if an entity is missing
        num_joints = len(env.scene[self._hand_name].joint_names)
        if num_joints != NUM_HAND_JOINTS:
            raise ValueError(
                f"{self.name} expects {NUM_HAND_JOINTS} hand joints, entity "
                f"'{self._hand_name}' declares {num_joints}"
            )
        self._origins = env.scene.env_origins(self._device).to(dtype=torch.float32)
        self._init_hand_pose = torch.zeros(
            (env.num_envs, NUM_HAND_JOINTS), dtype=torch.float32, device=self._device
        )
        self._zero_vel = torch.zeros_like(self._init_hand_pose)

    @property
    def init_hand_pose(self) -> torch.Tensor:
        """Hand joint positions written at each env's latest reset."""
        return self._init_hand_pose

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any) -> None:
        del params
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device)
        reset_mask = env.event_term_mask_buffer.scatter(env_ids)

        rows = env.rng.integers(0, int(self._cache.shape[0]), (env.num_envs,))
        sample = self._cache[rows]
        hand = sample[:, :NUM_HAND_JOINTS]
        if self._joint_noise > 0.0:
            hand = hand + env.rng.uniform(
                -self._joint_noise, self._joint_noise, tuple(hand.shape), dtype=hand.dtype
            )
        root = torch.zeros((env.num_envs, 13), dtype=torch.float32, device=self._device)
        root[:, 0:3] = sample[:, 16:19] + self._origins
        root[:, 3:7] = sample[:, 19:23]

        env.scene[self._hand_name].write_joint_state_to_sim_mask(
            hand, self._zero_vel, reset_mask, joint_ids=self._hand_joint_ids
        )
        env.scene[self._object_name].write_root_state_tensor_to_sim(root, reset_mask)

        action_term = env.action_manager.get_term(self._action_name)
        set_target = getattr(action_term, "set_target", None)
        if not callable(set_target):
            raise TypeError(
                f"{self.name} requires action term '{self._action_name}' to expose set_target "
                "(IntegratedJointPositionAction)"
            )
        set_target(hand[env_ids], env_ids)
        self._init_hand_pose = torch.where(reset_mask.unsqueeze(1), hand, self._init_hand_pose)
