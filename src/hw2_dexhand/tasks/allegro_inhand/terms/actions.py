"""Integrated joint-position action for the Allegro hand.

The original (NpEnv) Allegro envs drive the hand with a *persistent* position target
that each step moves by ``scale * clip(action, -1, 1)`` and is clamped to the
joint range::

    target_t = clamp(target_{t-1} + scale * clip(a_t, -1, 1), lower, upper)

Neither UniLab built-in reproduces this: ``JointPositionAction`` maps the raw
action to an absolute target and ``RelativeJointPositionAction`` is relative to
the *measured* joint position. This term keeps the integrated target as state,
resets it to the keyframe pose (or to a pose handed over by a reset event such
as the grasp-cache reset), and exposes it so an observation term can feed it
back to the policy (the "previous targets" block of the original layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from unilab.base.torch_env import TorchEnv
from unilab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class IntegratedJointPositionActionCfg(BaseActionCfg):
    """Configuration for :class:`IntegratedJointPositionAction`."""

    def build(self, env: ManagerBasedRlEnv) -> IntegratedJointPositionAction:
        return IntegratedJointPositionAction(self, env)


class IntegratedJointPositionAction(BaseAction):
    """Accumulate scaled actions into a joint-range-clamped position target."""

    cfg: IntegratedJointPositionActionCfg

    def __init__(self, cfg: IntegratedJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        default = self._entity.data.default_joint_pos(self._device)[:, self._target_index]
        self._default_target = default.to(dtype=torch.float32).clone()

        limits = self._entity.data.soft_joint_pos_limits(self._device)
        if limits.ndim == 2:
            limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)
        limits = limits[:, self._target_index, :].to(dtype=torch.float32)
        self._lower = limits[..., 0].contiguous()
        self._upper = limits[..., 1].contiguous()

        self._target = self._default_target.clone()
        self._processed_actions.copy_(self._target)
        self._tensor_target = torch.empty_like(self._target)
        # Reset events run before the action manager's own reset, which would
        # otherwise restore the keyframe target. ``set_target`` therefore also
        # records a per-row override that ``reset`` applies and clears.
        self._reset_override = self._default_target.clone()
        self._has_override = torch.zeros((self.num_envs,), dtype=torch.bool, device=self._device)

    @property
    def target(self) -> torch.Tensor:
        """Current integrated position targets, shape ``(num_envs, action_dim)``."""
        return self._target

    def set_target(self, values: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Overwrite the integrated target now and across the pending manager reset."""
        values = values.to(device=self._device, dtype=torch.float32)
        if env_ids is None:
            self._target.copy_(values)
            self._processed_actions.copy_(values)
            self._reset_override.copy_(values)
            self._has_override.fill_(True)
        else:
            self._target[env_ids] = values
            self._processed_actions[env_ids] = values
            self._reset_override[env_ids] = values
            self._has_override[env_ids] = True

    def _apply_affine(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        destination.copy_(source).clamp_(-1.0, 1.0).mul_(self._scale).add_(self._target)
        torch.clamp(destination, min=self._lower, max=self._upper, out=destination)
        self._target.copy_(destination)
        self._processed_actions_finite_valid.copy_(torch.isfinite(destination).all())

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias_tensor.to(
            device=self._device, non_blocking=True
        ).index_select(1, self._target_index)
        torch.sub(self._processed_actions, encoder_bias, out=self._tensor_target)
        self._entity.set_joint_position_target(self._tensor_target, joint_ids=self._target_ids)
        if isinstance(self._env, TorchEnv) and isinstance(
            self._entity.data.control_buffer, torch.Tensor
        ):
            self._env._pending_control_finite_check = bool(self._processed_actions_finite_valid)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        target = torch.where(
            self._has_override.unsqueeze(1), self._reset_override, self._default_target
        )
        rows = slice(None) if env_ids is None else env_ids
        self._target[rows] = target[rows]
        self._processed_actions[rows] = target[rows]
        self._has_override[rows] = False
