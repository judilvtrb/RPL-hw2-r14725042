"""Object physics randomization that remembers what it sampled.

UniLab's built-in mass/friction/COM event terms write through the reset
transaction but do not expose the sampled values. The adaptation track needs
them as privileged information, so this term samples per-env scale factors
and offsets, writes them through the same entity APIs, and keeps the sampled
values in per-env buffers that :class:`object_physics_priv` reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from unilab.envs.mdp.events import resolve_env_ids
from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg

from hw2_dexhand.tasks.allegro_inhand.terms.observations import _device

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

_OBJECT = SceneEntityCfg("object")
_HAND = SceneEntityCfg("robot")
PRIV_DIM = 5  # mass scale, friction scale, COM offset xyz
HAND_GAIN_PRIV_DIM = 1  # kp scale


def _range(values: Any, label: str) -> tuple[float, float]:
    if not isinstance(values, Sequence) or isinstance(values, str) or len(values) != 2:
        raise TypeError(f"{label} must be [low, high]")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise ValueError(f"{label} must satisfy low <= high, got {values}")
    return low, high


class randomize_object_physics(ManagerTermBase):
    """Reset event: scale the object's mass and sliding friction, offset its COM.

    Params: ``asset_cfg`` (object entity with ``body_names`` and ``geom_names``),
    ``mass_scale_range`` (default ``[1, 1]``), ``friction_scale_range`` (default
    ``[1, 1]``), ``com_offset_range`` (default ``[0, 0]``, metres, each axis).
    Sampled values are exposed as ``mass_scale``, ``friction_scale`` and
    ``com_offset`` (per env) and as ``privileged`` (``(num_envs, 5)``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _OBJECT)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._entity = env.scene[asset_cfg.name]
        self._device = _device(env)
        self._mass_range = _range(
            cfg.params.get("mass_scale_range", [1.0, 1.0]), "mass_scale_range"
        )
        self._friction_range = _range(
            cfg.params.get("friction_scale_range", [1.0, 1.0]), "friction_scale_range"
        )
        self._com_range = _range(cfg.params.get("com_offset_range", [0.0, 0.0]), "com_offset_range")

        self._body_ids, default_mass = self._entity.bind_body_mass_write(
            asset_cfg.body_ids, term_name=self.name
        )
        self._default_mass = default_mass.to(device=self._device, dtype=torch.float32)
        self._geom_ids, default_friction = self._entity.bind_geom_friction_write(
            asset_cfg.geom_ids, term_name=self.name
        )
        self._default_friction = default_friction.to(device=self._device, dtype=torch.float32)
        self._ipos_ids, default_ipos = self._entity.bind_body_ipos_write(
            asset_cfg.body_ids, term_name=self.name
        )
        self._default_ipos = default_ipos.to(device=self._device, dtype=torch.float32)

        num_envs = env.num_envs
        self.mass_scale = torch.ones((num_envs,), dtype=torch.float32, device=self._device)
        self.friction_scale = torch.ones((num_envs,), dtype=torch.float32, device=self._device)
        self.com_offset = torch.zeros((num_envs, 3), dtype=torch.float32, device=self._device)

    @property
    def privileged(self) -> torch.Tensor:
        """``(num_envs, 5)``: mass scale, friction scale, COM offset xyz."""
        return torch.cat(
            [self.mass_scale.unsqueeze(1), self.friction_scale.unsqueeze(1), self.com_offset], dim=1
        )

    def _rows(self, defaults: torch.Tensor, count: int) -> torch.Tensor:
        return defaults.unsqueeze(0).expand(count, *defaults.shape).clone()

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any) -> None:
        del params
        ids = resolve_env_ids(env, env_ids).to(self._device)
        count = int(ids.numel())
        if count == 0:
            return
        rng = env.rng
        mass_scale = rng.uniform(*self._mass_range, (count,), dtype=torch.float32).to(self._device)
        friction_scale = rng.uniform(*self._friction_range, (count,), dtype=torch.float32).to(
            self._device
        )
        com_offset = rng.uniform(*self._com_range, (count, 3), dtype=torch.float32).to(self._device)

        mass = self._rows(self._default_mass, count) * mass_scale.unsqueeze(1)
        friction = self._rows(self._default_friction, count)
        friction[..., 0] = friction[..., 0] * friction_scale.view(
            count, *([1] * (friction.ndim - 2))
        )
        ipos = self._rows(self._default_ipos, count) + com_offset.view(count, 1, 3)

        self._entity.write_body_mass_to_sim(
            mass, body_ids=self._body_ids, env_ids=ids, term_name=self.name
        )
        self._entity.write_geom_friction_to_sim(friction, self._geom_ids, ids, term_name=self.name)
        self._entity.write_body_ipos_to_sim(
            ipos, body_ids=self._ipos_ids, env_ids=ids, term_name=self.name
        )

        self.mass_scale[ids] = mass_scale
        self.friction_scale[ids] = friction_scale
        self.com_offset[ids] = com_offset


class object_physics_priv(ManagerTermBase):
    """Privileged observation: the object physics sampled by ``randomize_object_physics``.

    Params: ``event_name`` (default ``randomize_object_physics``). Returns
    ``(num_envs, 5)``: mass scale, friction scale, COM offset xyz. Intended for
    the critic group; the privileged baseline reads it through ``obs_groups.actor: [critic]``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._event_name = str(cfg.params.get("event_name", "randomize_object_physics"))
        event = env.event_manager.get_term_cfg(self._event_name).func
        if not isinstance(event, randomize_object_physics):
            raise TypeError(
                f"{self.name} requires event '{self._event_name}' to be randomize_object_physics"
            )
        self._event = event

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del env, params
        return self._event.privileged


__all__ = [
    "HAND_GAIN_PRIV_DIM",
    "PRIV_DIM",
    "hand_gains_priv",
    "object_physics_priv",
    "randomize_hand_gains",
    "randomize_object_physics",
]


class randomize_hand_gains(ManagerTermBase):
    """Reset event: scale the hand's actuator position gain (kp) per env.

    Params: ``asset_cfg`` (default entity ``robot``), ``kp_scale_range``
    (default ``[1, 1]``). The Allegro position actuators carry no velocity
    gain, so kd is written back unchanged. The sampled scale is exposed as
    ``kp_scale`` / ``privileged`` (``(num_envs, 1)``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        asset_cfg = cfg.params.get("asset_cfg", _HAND)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        self._entity = env.scene[asset_cfg.name]
        self._device = _device(env)
        self._kp_range = _range(cfg.params.get("kp_scale_range", [1.0, 1.0]), "kp_scale_range")
        actuator_ids = getattr(asset_cfg, "actuator_ids", None)
        self._actuator_ids, default_kp, default_kd = self._entity.bind_actuator_gain_write(
            actuator_ids if actuator_ids is not None else slice(None), term_name=self.name
        )
        self._default_kp = default_kp.to(device=self._device, dtype=torch.float32)
        self._default_kd = default_kd.to(device=self._device, dtype=torch.float32)
        self.kp_scale = torch.ones((env.num_envs,), dtype=torch.float32, device=self._device)

    @property
    def privileged(self) -> torch.Tensor:
        return self.kp_scale.unsqueeze(1)

    def _rows(self, defaults: torch.Tensor, count: int) -> torch.Tensor:
        return defaults.unsqueeze(0).expand(count, *defaults.shape).clone()

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params: Any) -> None:
        del params
        ids = resolve_env_ids(env, env_ids).to(self._device)
        count = int(ids.numel())
        if count == 0:
            return
        kp_scale = env.rng.uniform(*self._kp_range, (count,), dtype=torch.float32).to(self._device)
        kp = self._rows(self._default_kp, count) * kp_scale.view(
            count, *([1] * self._default_kp.ndim)
        )
        kd = self._rows(self._default_kd, count)
        self._entity.write_actuator_gains_to_sim(
            kp, kd, actuator_ids=self._actuator_ids, env_ids=ids, term_name=self.name
        )
        self.kp_scale[ids] = kp_scale


class hand_gains_priv(ManagerTermBase):
    """Privileged observation: the kp scale sampled by ``randomize_hand_gains``.

    Params: ``event_name`` (default ``randomize_hand_gains``).
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._event_name = str(cfg.params.get("event_name", "randomize_hand_gains"))
        event = env.event_manager.get_term_cfg(self._event_name).func
        if not isinstance(event, randomize_hand_gains):
            raise TypeError(
                f"{self.name} requires event '{self._event_name}' to be randomize_hand_gains"
            )
        self._event = event

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        del env, params
        return self._event.privileged
