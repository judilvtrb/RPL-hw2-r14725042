"""Config ownership: student overlay on top of the evaluator-owned reference.

A student config is a plain YAML overlay (no Hydra ``defaults``) with the same
structure as the owner YAML. ``hw2 train <cell> --config my.yaml`` composes
the cell's reference owner, merges the overlay, and refuses any key still set
to ``???``. At grading time the evaluator composes the same thing and then
re-imposes the evaluator-owned sections from the reference, so a submission
cannot move the goal posts; what it keeps from the student is what defines the
policy's I/O and training (observations, actions, rewards, training events,
``algo``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from hw2_dexhand.cli import compose_config
from hw2_dexhand.eval.tasks import ADAPT_BLINDFOLD_OVERRIDES, ADAPT_GROUP, TaskSpec

# Dotted paths the evaluator owns. Everything else in the composed config may
# be changed by the student.
EVALUATOR_OWNED: tuple[str, ...] = (
    "training.task_name",
    "training.sim_backend",
    "env.scene",
    "env.sim_dt",
    "env.ctrl_dt",
    "env.max_episode_seconds",
    "env.commands",
    "env.terminations",
    # All events: the reset events, the adaptation track's physics ranges, and
    # no student-added event. Training-time randomization a student adds
    # applies to training only; grading runs the reference events.
    "env.events",
    "env.policy_observation_group",
    "env.critic_observation_group",
)

# Overrides every evaluation run applies on top of the composed config.
EVAL_OVERRIDES: tuple[str, ...] = (
    "training.no_play=true",
    "training.device=cpu",
)

# Adaptation track: the training-time randomization of the four hidden
# properties is part of the grading contract (grading draws them from a wider,
# held-out range), so neither an overlay nor a command-line override may move
# it. ``compose_target`` refuses the config instead of silently re-imposing it.
ADAPT_TRAINING_LOCKED: tuple[str, ...] = (
    "env.events.randomize_object_physics",
    "env.events.randomize_hand_gains",
)
# Observation terms that return the sampled physics. They may feed the critic
# or any training-time module, never the graded actor.
PRIVILEGED_TERM_FUNCS: frozenset[str] = frozenset(
    {
        "hw2_dexhand.tasks.allegro_inhand.terms.object_physics_priv",
        "hw2_dexhand.tasks.allegro_inhand.terms.hand_gains_priv",
        "hw2_dexhand.tasks.allegro_inhand.terms.randomization.object_physics_priv",
        "hw2_dexhand.tasks.allegro_inhand.terms.randomization.hand_gains_priv",
    }
)
# Observation groups the graded actor may read on the multi and adaptation tracks.
GRADED_ACTOR_GROUPS: frozenset[str] = frozenset({"actor", "policy"})
# Modules the graded actor's observation terms must come from on those tracks.
# The packaged terms read joint state, object pose and velocities, sensors and
# the goal; none returns object identity or the sampled physics. Anything else
# the actor needs is computed inside the actor class, which only sees the
# observation tensor.
GRADED_TERM_MODULES: tuple[str, ...] = (
    "hw2_dexhand.tasks.allegro_inhand.terms.",
    "unilab.envs.mdp.",
)
# Multi-object contract: the policy never observes the object orientation, in
# the packaged ``object_quat`` term (dropped at grading) or under another name.
MULTI_FORBIDDEN_TERM_FUNCS: frozenset[str] = frozenset(
    {
        "hw2_dexhand.tasks.allegro_inhand.terms.object_quat_w",
        "hw2_dexhand.tasks.allegro_inhand.terms.object_quat_canonical_w",
        "hw2_dexhand.tasks.allegro_inhand.terms.active_object_quat_w",
        "hw2_dexhand.tasks.allegro_inhand.terms.observations.object_quat_w",
        "hw2_dexhand.tasks.allegro_inhand.terms.observations.object_quat_canonical_w",
        "hw2_dexhand.tasks.allegro_inhand.terms.multi_object.active_object_quat_w",
    }
)


_ABSENT = object()


class StudentConfigError(ValueError):
    """A student overlay is unusable (missing file, unfilled ???, bad structure)."""


def load_overlay(path: str | Path) -> DictConfig:
    overlay_path = Path(path)
    if not overlay_path.is_file():
        raise StudentConfigError(f"student config not found: {overlay_path}")
    loaded = OmegaConf.load(overlay_path)
    if not isinstance(loaded, DictConfig):
        raise StudentConfigError(f"student config must be a YAML mapping: {overlay_path}")
    if "defaults" in loaded:
        raise StudentConfigError(
            f"{overlay_path.name} must be a plain overlay without a Hydra 'defaults' list; "
            "the cell's reference owner is composed for you"
        )
    return loaded


def compose_reference(spec: TaskSpec, extra_overrides: Sequence[str] = ()) -> DictConfig:
    """The cell's reference config (owner + goal overrides), as `hw2 train <cell>` composes it."""
    return compose_config("ppo", spec.owner, [*spec.goal_overrides(), *extra_overrides])


def compose_eval_reference(spec: TaskSpec) -> DictConfig:
    """The reference plus the track's evaluator-owned grading overrides."""
    cfg = compose_reference(spec)
    overrides = spec.eval_overrides()
    if overrides:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        assert isinstance(cfg, DictConfig)
    return cfg


def missing_keys(cfg: DictConfig) -> list[str]:
    """Dotted keys whose value is still ``???``."""
    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, DictConfig):
            for key in node.keys():
                dotted = f"{prefix}.{key}" if prefix else str(key)
                if OmegaConf.is_missing(node, key):
                    found.append(dotted)
                else:
                    walk(node[key], dotted)

    walk(cfg, "")
    return found


def compose_target(
    target: TaskSpec | str,
    overlay: str | Path | None = None,
    extra_overrides: Sequence[str] = (),
) -> DictConfig:
    """Compose a graded cell or a packaged owner with an optional student overlay.

    Refuses an overlay with unfilled ``???``. For a cell, the goal overrides are
    applied last so an overlay cannot change them.
    """
    if isinstance(target, TaskSpec):
        cfg = compose_reference(target)
        final_overrides = target.goal_overrides()
    else:
        cfg = compose_config("ppo", str(target))
        final_overrides = []
    locked = (
        {dotted: _plain(_select(cfg, dotted)) for dotted in ADAPT_TRAINING_LOCKED}
        if _is_adapt(target)
        else None
    )
    # Students may add keys (new terms, actor kwargs); Hydra composes in struct mode.
    OmegaConf.set_struct(cfg, False)
    if overlay is not None:
        overlay_cfg = load_overlay(overlay)
        # Scan the overlay itself: OmegaConf.merge keeps the reference value
        # where the overlay says ???, so an unfilled template would otherwise
        # silently train with the packaged values.
        unfilled = missing_keys(overlay_cfg)
        if unfilled:
            raise StudentConfigError(
                f"{Path(overlay).name} still has {len(unfilled)} unfilled ??? field(s):\n  "
                + "\n  ".join(unfilled)
            )
        cfg = OmegaConf.merge(cfg, overlay_cfg)
        assert isinstance(cfg, DictConfig)
    if extra_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(extra_overrides)))
        assert isinstance(cfg, DictConfig)
    if final_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(final_overrides))
        assert isinstance(cfg, DictConfig)
    if locked is not None:
        moved = [k for k, v in locked.items() if _plain(_select(cfg, k)) != v]
        if moved:
            raise StudentConfigError(
                "the adaptation track trains under a fixed randomization range (see the "
                "packaged allegro_adapt owner); remove these keys from the overlay and the "
                "command line: " + ", ".join(moved)
            )
    return cfg


def _is_adapt(target: TaskSpec | str) -> bool:
    if isinstance(target, TaskSpec):
        return target.track == "adapt"
    return str(target).startswith(f"{ADAPT_GROUP}/")


def compose_student(
    spec: TaskSpec,
    overlay: str | Path | None,
    extra_overrides: Sequence[str] = (),
) -> DictConfig:
    """Reference cell config with the student's overlay merged; refuses unfilled ``???``."""
    return compose_target(spec, overlay, extra_overrides)


def _select(cfg: DictConfig, dotted: str) -> Any:
    return OmegaConf.select(cfg, dotted, default=None)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def ownership_differences(reference: DictConfig, candidate: DictConfig) -> list[str]:
    """Evaluator-owned dotted paths whose value differs between the two configs."""
    return [
        dotted
        for dotted in EVALUATOR_OWNED
        if _plain(_select(reference, dotted)) != _plain(_select(candidate, dotted))
    ]


def graded_actor_violations(cfg: DictConfig, track: str = "adapt") -> list[str]:
    """Ways a submitted config would let the graded actor read what the track hides.

    The adapter hands every observation group to the actor, and ``policy`` is
    the concatenation of every non-critic group, so both the group list and
    the term functions of every non-critic group are checked. Every actor term
    must come from a packaged term module; the adaptation track additionally
    forbids the sampled-physics terms and the multi-object track the
    object-orientation terms.
    """
    problems: list[str] = []
    groups = [str(g) for g in OmegaConf.select(cfg, "algo.obs_groups.actor", default=["policy"])]
    forbidden = sorted(g for g in groups if g not in GRADED_ACTOR_GROUPS)
    if forbidden:
        problems.append(
            f"algo.obs_groups.actor={groups} reads {forbidden}; the graded actor may only "
            f"read {sorted(GRADED_ACTOR_GROUPS)}"
        )
    critic_group = str(OmegaConf.select(cfg, "env.critic_observation_group", default="critic"))
    observations = OmegaConf.select(cfg, "env.observations", default=None)
    if OmegaConf.is_config(observations):
        for group_name, group in observations.items():
            if str(group_name) == critic_group or not OmegaConf.is_config(group):
                continue
            terms = OmegaConf.select(group, "terms", default=None)
            if not OmegaConf.is_config(terms):
                continue
            for term_name, term in terms.items():
                if not OmegaConf.is_config(term):
                    continue
                func = str(OmegaConf.select(term, "func", default=""))
                where = f"env.observations.{group_name}.terms.{term_name}"
                if track == "adapt" and func in PRIVILEGED_TERM_FUNCS:
                    problems.append(f"{where} returns the sampled physics ({func})")
                elif track == "multi" and func in MULTI_FORBIDDEN_TERM_FUNCS:
                    problems.append(f"{where} returns the object orientation ({func})")
                elif not func.startswith(GRADED_TERM_MODULES):
                    problems.append(f"{where} is not a packaged observation term ({func})")
    return problems


def compose_evaluation(
    spec: TaskSpec,
    overlay: str | Path | None,
    *,
    extra_overrides: Sequence[str] = (),
    allow_privileged: bool = False,
) -> tuple[DictConfig, list[str]]:
    """Student config with evaluator-owned sections re-imposed from the reference.

    Returns the config and the list of evaluator-owned paths the overlay had
    changed (reported as warnings; they are overwritten, not honoured). On the
    multi-object and adaptation tracks the graded actor's terms are checked
    after the track's grading overrides are applied (see
    :func:`graded_actor_violations`), and a config that fails is refused. On
    the adaptation track the evaluation env also drops the privileged terms
    altogether; ``allow_privileged`` keeps them, to measure a privileged
    reference policy, and is never used for grading.
    """
    reference = compose_eval_reference(spec)
    cfg = compose_student(spec, overlay)
    OmegaConf.set_struct(cfg, False)
    drift = ownership_differences(compose_reference(spec), cfg)
    # The student's training-only critic group (an asymmetric critic on a
    # track whose reference has none) is not part of the graded actor and may
    # read entities the evaluation scene lacks, so the evaluation env drops it.
    student_critic = _select(cfg, "env.critic_observation_group")
    reference_critic = _select(reference, "env.critic_observation_group")
    for dotted in EVALUATOR_OWNED:
        value = OmegaConf.select(reference, dotted, default=_ABSENT)
        if value is _ABSENT:
            continue
        # A null reference value (e.g. no critic group) is re-imposed as null.
        OmegaConf.update(cfg, dotted, value, merge=False)
    if student_critic and student_critic != reference_critic:
        OmegaConf.update(cfg, f"env.observations.{student_critic}", None, merge=False)
    overrides = [*spec.eval_overrides(), *EVAL_OVERRIDES]
    blind_adapt = spec.track == "adapt" and not allow_privileged
    if blind_adapt:
        overrides += list(ADAPT_BLINDFOLD_OVERRIDES)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([*overrides, *extra_overrides]))
    assert isinstance(cfg, DictConfig)
    # Checked after the overrides so the packaged ``object_quat`` term the
    # multi track drops is already gone; what remains is what the actor reads.
    if blind_adapt or spec.track == "multi":
        problems = graded_actor_violations(cfg, spec.track)
        if problems:
            hidden = "the sampled physics" if spec.track == "adapt" else "which object it holds"
            raise StudentConfigError(
                f"the graded actor may not read {hidden}: " + "; ".join(problems)
            )
    return cfg, drift


def build_env_cfg_override(cfg: DictConfig, root_dir: str | Path) -> dict[str, Any]:
    """Env override dict (reward + env sections) exactly as the PPO entrypoint builds it."""
    from unilab.base.config_adapter import EnvConfigAdapter

    return EnvConfigAdapter(cfg, root_dir=root_dir).build_task_env_cfg_override()


def create_env(cfg: DictConfig, *, num_envs: int, root_dir: str | Path) -> Any:
    from unilab.base import registry

    registry.ensure_registries()
    return registry.make(
        str(cfg.training.task_name),
        num_envs=num_envs,
        sim_backend=str(cfg.training.sim_backend),
        env_cfg_override=build_env_cfg_override(cfg, root_dir),
    )


__all__ = [
    "ADAPT_TRAINING_LOCKED",
    "EVALUATOR_OWNED",
    "EVAL_OVERRIDES",
    "GRADED_ACTOR_GROUPS",
    "GRADED_TERM_MODULES",
    "MULTI_FORBIDDEN_TERM_FUNCS",
    "PRIVILEGED_TERM_FUNCS",
    "StudentConfigError",
    "build_env_cfg_override",
    "compose_eval_reference",
    "compose_evaluation",
    "compose_reference",
    "compose_student",
    "compose_target",
    "create_env",
    "graded_actor_violations",
    "load_overlay",
    "missing_keys",
    "ownership_differences",
]
