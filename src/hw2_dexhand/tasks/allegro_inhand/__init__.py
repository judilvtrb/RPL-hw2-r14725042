"""Allegro in-hand task registrations.

Each task is a plain ``ManagerBasedRlEnvCfg`` whose manager terms are declared
in the owner YAML under ``hw2_dexhand/conf``; this module only tells the UniLab
registry which names exist and which backend builds them.
"""

from __future__ import annotations

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

HOLD_TASK_NAME = "AllegroInhandHold"
ROTATION_TASK_NAME = "AllegroInhandRotation"
REORIENT_TASK_NAME = "AllegroInhandReorient"
MULTI_TASK_NAME = "AllegroInhandMulti"
TASK_NAMES: tuple[str, ...] = (
    HOLD_TASK_NAME,
    ROTATION_TASK_NAME,
    REORIENT_TASK_NAME,
    MULTI_TASK_NAME,
)
SIM_BACKENDS: tuple[str, ...] = ("mujoco",)


def register_tasks() -> None:
    for name in TASK_NAMES:
        if registry.contains(name):
            continue
        registry.register_env_config(name, ManagerBasedRlEnvCfg)
        for backend in SIM_BACKENDS:
            registry.register_env(name, make_manager_based_rl_env, sim_backend=backend)


register_tasks()

__all__ = [
    "HOLD_TASK_NAME",
    "MULTI_TASK_NAME",
    "REORIENT_TASK_NAME",
    "ROTATION_TASK_NAME",
    "SIM_BACKENDS",
    "TASK_NAMES",
    "register_tasks",
]
