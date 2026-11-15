"""Goal-orientation command for in-hand reorientation.

The command is a unit quaternion ``(w, x, y, z)`` per env: the orientation the
object should reach. Goals are resampled when reached (not on a timer), and
the term counts the goals reached per episode as a metric, which UniLab logs
as ``Metrics/<term>/goals_reached`` and the evaluator reads directly.

Lifecycle, as UniLab orders it:

* ``reset`` (before the reset transaction commits): absolute-frame goals are
  sampled immediately; relative-frame goals store the sampled rotation and wait,
  because the object's post-reset pose is not readable yet.
* ``refresh_reset`` (after the commit and the read refresh): pending relative
  goals are composed with the post-reset object orientation.
* each step, after rewards: ``compute`` refreshes the error, counts successes,
  and resamples the goals of successful envs so the next observation sees them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unisim.backend.base import DebugPrimitive

from hw2_dexhand.tasks.allegro_inhand.quaternion import (
    quat_canonicalize,
    quat_error_magnitude,
    quat_mul,
    sample_uniform_quat,
    yaw_to_quat,
)
from hw2_dexhand.tasks.allegro_inhand.terms.object_view import ObjectView

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

GOAL_MODES: tuple[str, ...] = ("yaw", "so3")
GOAL_FRAMES: tuple[str, ...] = ("absolute", "relative")


@dataclass(kw_only=True)
class GoalOrientationCommandCfg(CommandTermCfg):
    """Goal distribution and success rule for a reorientation task.

    ``mode``: ``"yaw"`` samples a rotation about world z within ``yaw_range``;
    ``"so3"`` samples a uniformly random orientation. ``frame``: ``"absolute"``
    goals are world orientations about the object's canonical (identity) pose;
    ``"relative"`` goals compose the sampled rotation with the object's
    orientation at reset (and, on resample, at the previous success), so the
    policy must rotate the object *by* the sampled amount from where it is.
    """

    entity_name: str = "object"
    # Mixed pool: the entity packing all object bodies and their names; the
    # env's active object (env_id % len) is the one the goal applies to.
    body_names: tuple[str, ...] | list[str] | None = None
    mode: str = "yaw"
    frame: str = "absolute"
    yaw_range: tuple[float, float] | list[float] = (-math.pi, math.pi)
    success_tolerance_rad: float = 0.4
    resample_on_success: bool = True
    # 0 keeps episodes running after a success; otherwise the
    # ``goal_count_reached`` termination ends the episode at this count.
    max_consecutive_successes: int = 0
    # Goals change on success, never on the timer.
    resampling_time_range: tuple[float, float] | list[float] = (1.0e9, 1.0e9)
    overlay_axis_length: float = 0.08
    overlay_height_offset: float = 0.12

    def build(self, env: ManagerBasedRlEnv) -> GoalOrientationCommand:
        return GoalOrientationCommand(self, env)


class GoalOrientationCommand(CommandTerm):
    cfg: GoalOrientationCommandCfg

    def __init__(self, cfg: GoalOrientationCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.mode not in GOAL_MODES:
            raise ValueError(f"{self.name} mode must be one of {GOAL_MODES}, got {cfg.mode!r}")
        if cfg.frame not in GOAL_FRAMES:
            raise ValueError(f"{self.name} frame must be one of {GOAL_FRAMES}, got {cfg.frame!r}")
        yaw_range = tuple(float(v) for v in cfg.yaw_range)
        if len(yaw_range) != 2 or yaw_range[0] > yaw_range[1]:
            raise ValueError(f"{self.name} yaw_range must be [low, high] with low <= high")
        if float(cfg.success_tolerance_rad) <= 0.0:
            raise ValueError(f"{self.name} success_tolerance_rad must be positive")
        if int(cfg.max_consecutive_successes) < 0:
            raise ValueError(f"{self.name} max_consecutive_successes must be >= 0")
        self._yaw_range = yaw_range
        self._tolerance = float(cfg.success_tolerance_rad)
        params: dict[str, Any] = {"entity_name": cfg.entity_name}
        if cfg.body_names:
            params["tensor_body_names"] = list(cfg.body_names)
        else:
            from unilab.managers.scene_entity_config import SceneEntityCfg

            params["asset_cfg"] = SceneEntityCfg(cfg.entity_name)
        self._object = ObjectView(env, params, self.name)
        self._entity_name = self._object.entity_name
        self._origins = env.scene.env_origins(self._device).to(dtype=torch.float32)

        num_envs = self.num_envs
        self._goal = torch.zeros((num_envs, 4), dtype=torch.float32, device=self._device)
        self._goal[:, 0] = 1.0
        self._pending_relative = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)
        self._error = torch.zeros((num_envs,), dtype=torch.float32, device=self._device)
        self._success = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)
        self.metrics = {
            "goals_reached": torch.zeros((num_envs,), dtype=torch.float32, device=self._device),
            "orientation_error": torch.zeros((num_envs,), dtype=torch.float32, device=self._device),
            "success": torch.zeros((num_envs,), dtype=torch.float32, device=self._device),
        }

    # -- read-plan declaration -------------------------------------------------
    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self._object.body_names

    @property
    def entity_name(self) -> str:
        return self._entity_name

    # -- public state ----------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self._goal

    @property
    def goals_reached(self) -> torch.Tensor:
        """Goals reached in the current episode, per env (float counter)."""
        return self.metrics["goals_reached"]

    @property
    def error(self) -> torch.Tensor:
        return self._error

    @property
    def success(self) -> torch.Tensor:
        return self._success

    @property
    def success_tolerance_rad(self) -> float:
        return self._tolerance

    @property
    def max_consecutive_successes(self) -> int:
        return int(self.cfg.max_consecutive_successes)

    # -- helpers ---------------------------------------------------------------
    def current_quat(self) -> torch.Tensor | None:
        quat = self._object.field(self._env, "quat_w")
        return None if quat is None else quat.to(dtype=torch.float32)

    def current_pos(self) -> torch.Tensor | None:
        pos = self._object.field(self._env, "pos_w")
        return None if pos is None else pos.to(dtype=torch.float32)

    def _sample_rotation(self) -> torch.Tensor:
        if self.cfg.mode == "yaw":
            yaw = self._rng.uniform(
                self._yaw_range[0], self._yaw_range[1], (self.num_envs,), dtype=torch.float32
            )
            return yaw_to_quat(yaw.to(self._device))
        return sample_uniform_quat(self.num_envs, rng=self._rng, device=self._device)

    def _assign_goals(self, mask: torch.Tensor, reference: torch.Tensor | None) -> None:
        rotation = self._sample_rotation()
        if self.cfg.frame == "relative":
            if reference is None:
                # Keep the raw rotation; refresh_reset composes it with the
                # post-reset object orientation.
                self._goal = torch.where(mask.unsqueeze(1), rotation, self._goal)
                self._pending_relative |= mask
                return
            rotation = quat_mul(rotation, reference)
        self._goal = torch.where(mask.unsqueeze(1), quat_canonicalize(rotation), self._goal)
        self._pending_relative &= ~mask

    def _finalize_pending(self) -> None:
        if not bool(self._pending_relative.any()):
            return
        current = self.current_quat()
        if current is None:
            return
        composed = quat_canonicalize(quat_mul(self._goal, current))
        self._goal = torch.where(self._pending_relative.unsqueeze(1), composed, self._goal)
        self._pending_relative.zero_()

    def _refresh_error(self, *, update_success: bool = True) -> None:
        current = self.current_quat()
        if current is None:
            return
        self._error = quat_error_magnitude(self._goal, current)
        if update_success:
            self._success = self._error <= self._tolerance
            self.metrics["success"].copy_(self._success.to(torch.float32))
        self.metrics["orientation_error"].copy_(self._error)

    # -- CommandTerm hooks -----------------------------------------------------
    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        # Called from reset() before the reset transaction commits (object
        # reads are stale) and from the never-expiring timer.
        self._assign_goals(reset_mask, reference=None)
        self._success.masked_fill_(reset_mask, False)
        self._error.masked_fill_(reset_mask, 0.0)

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask
        self._finalize_pending()
        self._refresh_error()

    def post_compute_reset(self) -> None:
        pass

    def _update_metrics(self) -> None:
        self._refresh_error()

    def _update_command(self) -> None:
        self._finalize_pending()
        success = self._success
        self.metrics["goals_reached"].add_(success.to(torch.float32))
        if self.cfg.resample_on_success and bool(success.any()):
            self._assign_goals(success, reference=self.current_quat())
            # Keep ``success`` as the flag that was counted this step (readers
            # such as the evaluator see one True per goal reached); only the
            # error tracks the freshly sampled goal.
            self._refresh_error(update_success=False)

    # -- playback overlay ------------------------------------------------------
    def playback_debug_overlay_getter(self):
        axis_length = float(self.cfg.overlay_axis_length)
        height_offset = float(self.cfg.overlay_height_offset)

        def getter():
            current = self.current_pos()
            if current is None:
                return None
            pos = (current - self._origins).detach().cpu()
            goal = self._goal.detach().cpu()
            frames = []
            for row in range(self.num_envs):
                x, y, z = pos[row].tolist()
                frames.append(
                    [
                        DebugPrimitive(
                            kind="frame",
                            pos=(x, y, z + height_offset),
                            quat=tuple(goal[row].tolist()),
                            size=(axis_length,),
                        )
                    ]
                )
            return frames

        return getter


__all__ = ["GOAL_FRAMES", "GOAL_MODES", "GoalOrientationCommand", "GoalOrientationCommandCfg"]
