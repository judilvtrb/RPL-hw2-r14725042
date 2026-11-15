"""Checkpoint resolution and policy loading through the UniLab/rsl_rl runner."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


class CheckpointError(ValueError):
    """The checkpoint cannot be found, read, or does not fit the configured policy."""


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a ``model_*.pt`` file or a run directory (its highest-numbered model)."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        models = sorted(
            candidate.glob("model_*.pt"),
            key=lambda item: int(item.stem.split("_", 1)[1]) if item.stem[6:].isdigit() else -1,
        )
        if models:
            return models[-1]
        raise CheckpointError(f"no model_*.pt in run directory {candidate}")
    raise CheckpointError(f"checkpoint not found: {candidate}")


@dataclass(frozen=True)
class LoadedPolicy:
    checkpoint: Path
    policy_obs_mode: str
    wrapped_env: Any
    policy: Callable[[Any], torch.Tensor]


def load_policy(
    cfg: DictConfig,
    env: Any,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> LoadedPolicy:
    """Wrap ``env`` for rsl_rl, build the runner from ``cfg.algo``, load the checkpoint.

    The network comes from ``cfg.algo`` (the student's config); a mismatch with
    the checkpoint, or between the checkpoint input width and the env
    observation width, is reported as :class:`CheckpointError`.
    """
    from rsl_rl.runners import OnPolicyRunner
    from unilab.rl import RslRlVecEnvAdapter, get_policy_obs_dims
    from unilab.training.run import algo_config_dict
    from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

    ckpt_path = resolve_checkpoint(checkpoint)
    actor_obs_dim, flat_obs_dim = get_policy_obs_dims(env.obs_groups_spec)
    critic_dim = int(env.obs_groups_spec.get("critic", actor_obs_dim))
    actor_groups = [
        str(g) for g in OmegaConf.select(cfg, "algo.obs_groups.actor", default=["policy"])
    ]
    policy_obs_mode = (
        "flat" if "policy" in actor_groups and flat_obs_dim != actor_obs_dim else "actor"
    )
    widths = {
        "actor": actor_obs_dim,
        "policy": flat_obs_dim if policy_obs_mode == "flat" else actor_obs_dim,
        "critic": critic_dim,
    }
    unknown = [g for g in actor_groups if g not in widths]
    if unknown:
        raise CheckpointError(f"algo.obs_groups.actor names unknown groups {unknown}")
    expected = sum(widths[g] for g in actor_groups)
    # The width heuristic reads the first MLP layer, which equals the observation
    # width only for rsl_rl's stock MLPModel; custom actor classes are validated
    # by the strict state-dict load below instead.
    actor_class = str(
        OmegaConf.select(cfg, "algo.actor.class_name", default="rsl_rl.models.MLPModel")
    )
    ckpt_dim = (
        infer_checkpoint_actor_input_dim(str(ckpt_path))
        if actor_class == "rsl_rl.models.MLPModel"
        else None
    )
    if ckpt_dim is not None and ckpt_dim != expected:
        raise CheckpointError(
            f"{ckpt_path.name} expects {ckpt_dim} actor inputs but the evaluation env "
            f"provides {expected} for algo.obs_groups.actor={actor_groups} "
            f"(actor {actor_obs_dim}, flat {flat_obs_dim}, critic {critic_dim}). The submitted "
            "config must declare the observation terms and obs_groups the policy was trained with."
        )

    wrapped_env = RslRlVecEnvAdapter(env, device=device, policy_obs_mode=policy_obs_mode)
    train_cfg = copy.deepcopy(algo_config_dict(cfg))
    train_cfg.setdefault("runner", {})["logger"] = "none"
    # Only the actor is evaluated. A submission's critic may have been trained
    # with a critic group this env does not provide (e.g. privileged physics):
    # build a placeholder critic on the actor groups and load the actor alone.
    if "critic" not in env.obs_groups_spec:
        train_cfg.setdefault("obs_groups", {})["critic"] = list(actor_groups)
    load_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
    }
    try:
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
        runner.load(str(ckpt_path), load_cfg=load_cfg, map_location=device)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise CheckpointError(
            f"cannot load {ckpt_path} with the submitted algo config "
            f"(actor {train_cfg.get('actor')}): {exc}"
        ) from exc
    policy = runner.get_inference_policy(device=device)
    return LoadedPolicy(
        checkpoint=ckpt_path,
        policy_obs_mode=policy_obs_mode,
        wrapped_env=wrapped_env,
        policy=policy,
    )


__all__ = ["CheckpointError", "LoadedPolicy", "load_policy", "resolve_checkpoint"]
