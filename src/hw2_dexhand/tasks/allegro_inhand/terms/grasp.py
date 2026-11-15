"""Terms for grasp-cache collection (``hw2 make-caches``).

Collection drives the hold scene with zero actions from the keyframe pose plus
joint noise, lets the hand settle for a few seconds, and keeps the states that
still hold the object with enough fingertip contact. These terms provide the
noisy reset and the measurements the collector needs, as observation terms in
a dedicated group so nothing else changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from unilab.envs.mdp.observations import joint_view
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.object_view import device_of, ready_read_plan

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

_HAND = SceneEntityCfg("robot")
FINGERTIP_BODIES: tuple[str, ...] = ("ff_tip", "mf_tip", "rf_tip", "th_tip")


class reset_joint_noise(ManagerTermBase):
    """Reset event: hand joints = keyframe pose + uniform ``±joint_noise`` (rad).

    Params: ``joint_noise``, ``asset_cfg`` (default ``robot``), ``action_name``
    (default ``hand``; the integrated target is seeded with the noisy pose).
    Runs after ``reset_scene_to_default`` so the object starts at its keyframe pose.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _HAND)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._asset_cfg = asset_cfg
        self._device = device_of(env)
        self._joint_noise = float(cfg.params.get("joint_noise", 0.0))
        self._action_name = str(cfg.params.get("action_name", "hand"))
        self._entity = env.scene[asset_cfg.name]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any) -> None:
        del params
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device)
        reset_mask = env.event_term_mask_buffer.scatter(env_ids)
        joint_ids = self._asset_cfg.joint_ids
        default = self._entity.data.default_joint_pos(self._device)[:, joint_ids].to(torch.float32)
        noise = env.rng.uniform(
            -self._joint_noise, self._joint_noise, tuple(default.shape), dtype=torch.float32
        ).to(self._device)
        limits = self._entity.data.soft_joint_pos_limits(self._device)
        if limits.ndim == 2:
            limits = limits.unsqueeze(0).expand(env.num_envs, -1, -1)
        limits = limits[:, joint_ids, :]
        pose = torch.clamp(default + noise, min=limits[..., 0], max=limits[..., 1])
        self._entity.write_joint_state_to_sim_mask(
            pose, torch.zeros_like(pose), reset_mask, joint_ids=joint_ids
        )
        set_target = getattr(env.action_manager.get_term(self._action_name), "set_target", None)
        if callable(set_target):
            set_target(pose[env_ids], env_ids)


class hand_joint_pos_raw(ManagerTermBase):
    """Unnormalized hand joint positions (what a grasp cache stores)."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _HAND)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name

    @property
    def tensor_state_entity(self) -> str:
        return self._entity_name

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        return joint_view(env, self._asset_cfg, self.name).joint_pos[:, self._asset_cfg.joint_ids]


class fingertip_pos_w(ManagerTermBase):
    """World positions of the four fingertip bodies, flattened to ``(N, 12)``.

    Params: ``entity_name`` (default ``robot``), ``body_names`` (default the
    Allegro fingertips).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._entity_name = str(cfg.params.get("entity_name", "robot"))
        names = cfg.params.get("body_names", FINGERTIP_BODIES)
        if not isinstance(names, Sequence) or isinstance(names, str) or not names:
            raise TypeError(f"{self.name} body_names must be a non-empty list")
        self._body_names = tuple(str(name) for name in names)
        self._device = device_of(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        read_plan = ready_read_plan(env)
        if read_plan is None:
            return torch.zeros(
                (env.num_envs, 3 * len(self._body_names)), dtype=torch.float32, device=self._device
            )
        view = read_plan.body_tensor_view(self._entity_name, self._body_names)
        return view.pos_w.reshape(env.num_envs, -1)


__all__ = ["FINGERTIP_BODIES", "fingertip_pos_w", "hand_joint_pos_raw", "reset_joint_noise"]
