"""Observation terms for the Allegro in-hand tasks.

UniLab's managers probe every observation term once at construction, before the
scene's packed tensor read plan exists, to size their buffers. Terms that read
body state therefore (a) declare the bodies they need through
``tensor_body_names`` / ``entity_name`` so the env packs them into the read
plan, and (b) return a correctly shaped zeros tensor while the plan is absent.
Class-based terms bind the entity on the cold path and keep the hot path to a
single packed-view read, mirroring UniLab's own task terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from unilab.envs.mdp.observations import joint_view
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.object_view import (
    ObjectView,
)
from hw2_dexhand.tasks.allegro_inhand.terms.object_view import (
    device_of as _device,
)

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

# The hand is the scene's primary entity, which UniLab's env factory requires
# to be named "robot" whenever a second floating entity (the object) exists.
_HAND = SceneEntityCfg("robot")
_OBJECT = SceneEntityCfg("object")


class hand_joint_pos_normalized(ManagerTermBase):
    """Joint positions mapped to ``[-1, 1]`` over each joint's position range.

    Declares ``tensor_state_entity`` so the env packs the hand's joint state
    into the read plan (upstream only does that for its own joint terms).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _HAND)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        device = _device(env)
        limits = env.scene[self._entity_name].data.soft_joint_pos_limits(device)
        if limits.ndim == 2:
            limits = limits.unsqueeze(0).expand(env.num_envs, -1, -1)
        limits = limits[:, asset_cfg.joint_ids, :].to(dtype=torch.float32)
        self._mid = (0.5 * (limits[..., 0] + limits[..., 1])).contiguous()
        self._inv_half_range = (1.0 / (0.5 * (limits[..., 1] - limits[..., 0]) + 1e-8)).contiguous()

    @property
    def tensor_state_entity(self) -> str:
        return self._entity_name

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _HAND) -> torch.Tensor:
        del asset_cfg
        state = joint_view(env, self._asset_cfg, self.name)
        return (state.joint_pos[:, self._asset_cfg.joint_ids] - self._mid) * self._inv_half_range


class joint_position_target(ManagerTermBase):
    """The integrated position targets held by the named action term.

    Params: ``action_name`` (default ``hand``), ``entity_name`` (default
    ``robot``; sizes the probe before the action manager exists).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._action_name = str(cfg.params.get("action_name", "hand"))
        self._entity_name = str(cfg.params.get("entity_name", "robot"))
        self._width = len(env.scene[self._entity_name].joint_names)
        self._device = _device(env)

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def __call__(
        self, env: ManagerBasedRlEnv, action_name: str = "hand", entity_name: str = "robot"
    ) -> torch.Tensor:
        del action_name, entity_name
        action_manager = getattr(env, "action_manager", None)
        if action_manager is None:
            return torch.zeros(
                (env.num_envs, self._width), dtype=torch.float32, device=self._device
            )
        term = action_manager.get_term(self._action_name)
        target = getattr(term, "target", None)
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"joint_position_target requires action term '{self._action_name}' to expose "
                "an integrated 'target' tensor (IntegratedJointPositionAction)"
            )
        return target


class _ObjectRootTerm(ManagerTermBase):
    """Object term reading one ``(num_envs, k)`` field through :class:`ObjectView`.

    Single-object owners pass ``asset_cfg`` (root entity); the mixed pool passes
    ``entity_name`` + ``tensor_body_names`` and the env's active object is read.
    """

    _field: str = "pos_w"
    _width: int = 3

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._object = ObjectView(env, dict(cfg.params), self.name)
        self._entity_name = self._object.entity_name
        self._device = self._object.device

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._object.body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def _value(self, env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
        return self._object.field(env, name)

    def _view(self, env: ManagerBasedRlEnv) -> Any | None:  # kept for subclasses
        return self._object.view(env)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        values = self._value(env, self._field)
        if values is None:
            return torch.zeros(
                (env.num_envs, self._width), dtype=torch.float32, device=self._device
            )
        return values


class object_pos_w(_ObjectRootTerm):
    """World position of the object's root body."""

    _field = "pos_w"
    _width = 3


class object_quat_w(_ObjectRootTerm):
    """World orientation (w, x, y, z) of the object's root body."""

    _field = "quat_w"
    _width = 4


class object_lin_vel_w(_ObjectRootTerm):
    """World linear velocity of the object's root body."""

    _field = "lin_vel_w"
    _width = 3


class object_ang_vel_w(_ObjectRootTerm):
    """World angular velocity of the object's root body."""

    _field = "ang_vel_w"
    _width = 3


class object_quat_canonical_w(object_quat_w):
    """Canonical (``w >= 0``) world orientation of the object's root body."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        from hw2_dexhand.tasks.allegro_inhand.quaternion import quat_canonicalize

        return quat_canonicalize(super().__call__(env, **params))


class goal_relative_quat(_ObjectRootTerm):
    """Rotation still needed to reach the goal, ``canonical(goal * inv(current))``.

    Params: ``command_name`` (default ``goal``), ``asset_cfg`` (the object).
    """

    _width = 4

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = str(cfg.params.get("command_name", "goal"))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        quat = self._value(env, "quat_w")
        if quat is None:
            probe = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=self._device)
            probe[:, 0] = 1.0
            return probe
        from hw2_dexhand.tasks.allegro_inhand.quaternion import (
            quat_canonicalize,
            quat_inv,
            quat_mul,
        )

        goal = env.command_manager.get_term(self._command_name).command
        return quat_canonicalize(quat_mul(goal, quat_inv(quat)))
