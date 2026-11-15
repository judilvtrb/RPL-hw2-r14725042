"""Reward terms for the Allegro in-hand rotation task (ported from the original NpEnv env).

Every term returns an unweighted per-env value; the owner YAML sets the weight
and ``scale_rewards_by_dt`` multiplies the weighted sum by ``ctrl_dt`` as the
original env did.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from unilab.envs.mdp.observations import joint_view
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.observations import _device, _ObjectRootTerm

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

_HAND = SceneEntityCfg("robot")


def _vec3(values: Any, label: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, str) or len(values) != 3:
        raise TypeError(f"{label} must be a list of 3 numbers")
    return [float(v) for v in values]


class object_angvel_about_axis(_ObjectRootTerm):
    """Object angular velocity projected on ``axis`` (world frame), clipped.

    Params: ``axis`` (default ``[0, 0, 1]``), ``clip`` (default ``[-0.5, 0.5]``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        axis = torch.tensor(
            _vec3(cfg.params.get("axis", [0.0, 0.0, 1.0]), f"{self.name} axis"),
            dtype=torch.float32,
            device=self._device,
        )
        self._axis = axis / torch.linalg.norm(axis).clamp_min(1e-8)
        clip = cfg.params.get("clip", [-0.5, 0.5])
        if not isinstance(clip, Sequence) or len(clip) != 2 or float(clip[0]) > float(clip[1]):
            raise ValueError(f"{self.name} clip must be [low, high] with low <= high")
        self._clip = (float(clip[0]), float(clip[1]))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        ang_vel = self._value(env, "ang_vel_w")
        if ang_vel is None:
            return torch.zeros((env.num_envs,), dtype=torch.float32, device=self._device)
        return (ang_vel @ self._axis).clamp(*self._clip)


class object_lin_vel_l1(_ObjectRootTerm):
    """Sum of absolute object linear velocity components (a penalty)."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        lin_vel = self._value(env, "lin_vel_w")
        if lin_vel is None:
            return torch.zeros((env.num_envs,), dtype=torch.float32, device=self._device)
        return lin_vel.abs().sum(dim=1)


class _HandJointTerm(ManagerTermBase):
    """Cold-path binding to the hand's joint state."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _HAND)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._asset_cfg = asset_cfg
        self._entity_name = asset_cfg.name
        self._device = _device(env)

    @property
    def tensor_state_entity(self) -> str:
        return self._entity_name

    @property
    def entity_name(self) -> str:
        return self._entity_name

    def _joints(self, env: ManagerBasedRlEnv) -> Any:
        return joint_view(env, self._asset_cfg, self.name)


class hand_pose_diff_from_reset(_HandJointTerm):
    """Squared joint distance from the pose written at the last grasp-cache reset.

    Params: ``event_name`` (default ``reset_from_grasp_cache``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._event_name = str(cfg.params.get("event_name", "reset_from_grasp_cache"))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        event = env.event_manager.get_term_cfg(self._event_name).func
        init_pose = getattr(event, "init_hand_pose", None)
        if not isinstance(init_pose, torch.Tensor):
            raise TypeError(
                f"{self.name} requires event '{self._event_name}' to expose init_hand_pose "
                "(reset_from_grasp_cache)"
            )
        state = self._joints(env)
        diff = state.joint_pos[:, self._asset_cfg.joint_ids] - init_pose
        return diff.square().sum(dim=1)


class _PdTorqueTerm(_HandJointTerm):
    """Proxy PD torque ``kp (target - q) - kd qdot`` from the integrated targets.

    Params: ``kp`` (default 1.0), ``kd`` (default 0.1), ``torque_clip`` (default
    0.5, the original env's clamp), ``action_name`` (default ``hand``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._kp = float(cfg.params.get("kp", 1.0))
        self._kd = float(cfg.params.get("kd", 0.1))
        self._torque_clip = float(cfg.params.get("torque_clip", 0.5))
        self._action_name = str(cfg.params.get("action_name", "hand"))

    def _torque_and_vel(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
        target = getattr(env.action_manager.get_term(self._action_name), "target", None)
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"{self.name} requires action term '{self._action_name}' to expose target"
            )
        state = self._joints(env)
        joint_ids = self._asset_cfg.joint_ids
        torque = (
            self._kp * (target - state.joint_pos[:, joint_ids])
            - self._kd * state.joint_vel[:, joint_ids]
        )
        return torque.clamp(-self._torque_clip, self._torque_clip), state.joint_vel[:, joint_ids]


class hand_pd_torque_l2(_PdTorqueTerm):
    """Sum of squared proxy PD torques (a penalty)."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        torque, _ = self._torque_and_vel(env)
        return torque.square().sum(dim=1)


class hand_pd_work_sq(_PdTorqueTerm):
    """Squared mechanical work ``(sum tau * qdot)^2`` of the proxy PD torques."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        torque, joint_vel = self._torque_and_vel(env)
        return (torque * joint_vel).sum(dim=1).square()


class _GoalErrorTerm(_ObjectRootTerm):
    """Orientation error between the object and the goal command."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = str(cfg.params.get("command_name", "goal"))

    def _goal_term(self, env: ManagerBasedRlEnv) -> Any:
        return env.command_manager.get_term(self._command_name)

    def _error(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        from hw2_dexhand.tasks.allegro_inhand.quaternion import quat_error_magnitude

        quat = self._value(env, "quat_w")
        if quat is None:
            return None
        return quat_error_magnitude(self._goal_term(env).command, quat)


class goal_orientation_reward(_GoalErrorTerm):
    """Dense ``1 / (error + rot_eps)``; larger when closer to the goal."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._rot_eps = float(cfg.params.get("rot_eps", 0.1))
        if self._rot_eps <= 0.0:
            raise ValueError(f"{self.name} rot_eps must be positive")

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        error = self._error(env)
        if error is None:
            return torch.zeros((env.num_envs,), dtype=torch.float32, device=self._device)
        return 1.0 / (error.abs() + self._rot_eps)


class goal_success(_GoalErrorTerm):
    """1 for every step spent inside the goal command's success tolerance."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del params
        error = self._error(env)
        if error is None:
            return torch.zeros((env.num_envs,), dtype=torch.float32, device=self._device)
        tolerance = float(self._goal_term(env).success_tolerance_rad)
        return (error <= tolerance).to(torch.float32)
