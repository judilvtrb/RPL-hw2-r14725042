"""``hw2`` command line.

UniLab's own ``train`` CLI only sees configs inside the UniLab package, so HW2
composes Hydra itself (from ``hw2_dexhand/conf``) and calls the UniLab training
entrypoint with the composed config, the same way other UniLab task packages do.

Commands:

* ``hw2 list`` / ``hw2 cells``: packaged owners / the graded task cells.
* ``hw2 train <owner|cell> [--config overlay.yaml] [overrides]``.
* ``hw2 play  <owner|cell> [--config overlay.yaml] [overrides]`` (needs a run).
* ``hw2 eval  <cell> <checkpoint> [--config overlay.yaml] [--challenge public]``.
* ``hw2 grade <manifest.yaml ...>``: evaluate submissions and write JSON reports.
* ``hw2 template [--adapt] [PATH]``: the overlay templates.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import hw2_dexhand  # noqa: F401  (registers the hw2_assets / hw2_cache resolvers)
from hw2_dexhand import CONF_ROOT

SUPPORTED_ALGOS = ("ppo",)
# Intermediate YAMLs that other owners inherit from; not runnable on their own.
ABSTRACT_OWNERS = frozenset({"base", "single", "_adapt", "_adapt_priv"})


def compose_config(
    algo: str,
    task: str,
    overrides: Sequence[str] = (),
    *,
    config_name: str = "config",
) -> DictConfig:
    """Compose ``<conf>/<algo>/config.yaml`` with ``task=<task>`` plus overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algo {algo!r}; expected one of {SUPPORTED_ALGOS}")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_ROOT / algo), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=[f"task={task}", *overrides])
    # Resolve ${hw2_assets:...} now so the plain-dict env override and any
    # spawned worker see absolute asset paths rather than interpolations.
    OmegaConf.resolve(cfg)
    return cfg


def list_tasks(algo: str = "ppo") -> list[str]:
    """Return the packaged task owners as ``<task>/<owner>`` strings."""
    root = CONF_ROOT / algo / "task"
    return sorted(
        f"{path.parent.name}/{path.stem}"
        for path in root.glob("*/*.yaml")
        if path.stem not in ABSTRACT_OWNERS
    )


def _compose_target(target: str, config: str | None, overrides: Sequence[str]) -> DictConfig:
    """Compose a packaged owner, or a graded cell with an optional student overlay."""
    from hw2_dexhand.eval.config import compose_target
    from hw2_dexhand.eval.tasks import parse_cell

    if target in list_tasks("ppo"):
        if config is None:
            return compose_config("ppo", target, overrides)
        return compose_target(target, config, extra_overrides=overrides)
    try:
        spec = parse_cell(target)
    except ValueError:
        raise SystemExit(f"unknown task {target!r}; see `hw2 list` and `hw2 cells`") from None
    return compose_target(spec, config, extra_overrides=overrides)


def _run_training(cfg: DictConfig) -> None:
    from unilab.scripts import train_rsl_rl

    train_rsl_rl.main(cfg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hw2", description="HW2 dexterous-hand tasks on UniLab")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("train", "train a packaged owner or a graded cell (with --config)"),
        ("play", "replay a trained checkpoint (sets training.play_only=true)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("task", help="owner (hw2 list) or cell (hw2 cells)")
        p.add_argument("--config", default=None, help="student overlay YAML for a cell")
        p.add_argument("--cfg", action="store_true", help="print the composed config and exit")
        p.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. algo.num_envs=256")

    sub.add_parser("list", help="list packaged task owners")
    sub.add_parser("cells", help="list the graded task cells")
    p = sub.add_parser("template", help="print the student config template (or copy it to PATH)")
    p.add_argument("path", nargs="?", default=None)
    p.add_argument(
        "--adapt", action="store_true", help="the adaptation-track (problem 3) overlay template"
    )

    p = sub.add_parser("eval", help="evaluate one checkpoint on one cell")
    p.add_argument("cell")
    p.add_argument("checkpoint", help="model_*.pt file or run directory")
    p.add_argument("--config", default=None, help="the overlay the policy was trained with")
    _add_eval_args(p)
    p.add_argument("--out", default=None, help="write the JSON result here")

    p = sub.add_parser("make-caches", help="generate grasp caches by simulation")
    p.add_argument("--objects", nargs="*", default=None, help="default: ball cube cylinder")
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None, help="default: the package cache directory")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("grade", help="evaluate submission manifests")
    p.add_argument("manifests", nargs="+")
    _add_eval_args(p)
    p.add_argument("--cells", nargs="*", default=None, help="only these cells")
    p.add_argument("--out-dir", default=None, help="write one JSON report per manifest here")
    return parser


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=None, help="per cell (rubric default)")
    parser.add_argument("--num-envs", type=int, default=16, help="parallel envs per batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rubric", default=None, help="rubric YAML (package default)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--challenge",
        default=None,
        help="extra scenarios per cell: 'public' (the packaged set) or a challenge YAML",
    )
    parser.add_argument(
        "--allow-privileged",
        action="store_true",
        help="let an adaptation-track actor read the sampled physics (measures a "
        "privileged reference policy; never used for grading)",
    )


def _load_challenges(path: str | None):
    if path is None:
        return ()
    from hw2_dexhand.eval.challenges import load_challenges

    return tuple(load_challenges(path))


def _cmd_eval(args: argparse.Namespace) -> int:
    from hw2_dexhand.eval.evaluate import evaluate_cell, format_cell_line, write_json
    from hw2_dexhand.eval.scoring import Rubric
    from hw2_dexhand.eval.tasks import parse_cell

    rubric = Rubric.load(args.rubric)
    spec = parse_cell(args.cell)
    result = evaluate_cell(
        spec,
        args.checkpoint,
        config=args.config,
        episodes=args.episodes or rubric.default_episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        rubric=rubric,
        root_dir=Path.cwd(),
        challenges=_load_challenges(args.challenge),
        allow_privileged=bool(args.allow_privileged),
    )
    print(format_cell_line(result))
    for warning in result.warnings:
        print(f"  warning: {warning}")
    if args.out:
        write_json(result.to_dict(), args.out)
    return 0 if result.error is None else 1


def _cmd_grade(args: argparse.Namespace) -> int:
    from hw2_dexhand.eval.evaluate import evaluate_submission, write_json
    from hw2_dexhand.eval.scoring import Rubric

    rubric = Rubric.load(args.rubric)
    status = 0
    for manifest in args.manifests:
        report = evaluate_submission(
            manifest,
            episodes=args.episodes,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            rubric=rubric,
            cells=args.cells,
            root_dir=Path.cwd(),
            challenges=_load_challenges(args.challenge),
            allow_privileged=bool(args.allow_privileged),
        )
        print(f"== {report['team']}: {report['total_points']:.1f} / {report['max_points']:.0f}")
        for cell in report["cells"]:
            value = cell["score"]["value"]
            shown = "-" if value is None else f"{value:.2f}"
            tail = f"ERROR {cell['error']}" if cell["error"] else f"tier {cell['score']['tier']}"
            print(f"  {cell['cell']:30s} {shown:>6s}  {cell['score']['points']:.1f} pts  {tail}")
            for item in cell.get("challenges", []):
                v = item["score"]["value"]
                shown = "-" if v is None else f"{v:.2f}"
                tail = f"ERROR {item['error']}" if item["error"] else ""
                print(f"    {item['cell']:28s} {shown:>6s}  {tail}")
            if cell["error"]:
                status = 1
        if args.out_dir:
            write_json(report, Path(args.out_dir) / f"{report['team']}.json")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        for owner in list_tasks("ppo"):
            print(owner)
        return 0
    if args.command == "cells":
        from hw2_dexhand.eval.tasks import TASK_MATRIX

        for spec in TASK_MATRIX:
            print(f"{spec.key:28s} {spec.title}")
        return 0
    if args.command == "template":
        from hw2_dexhand import TEMPLATES_ROOT

        name = "adapt_config.yaml" if args.adapt else "student_config.yaml"
        template = TEMPLATES_ROOT / name
        if args.path is None:
            print(template.read_text(encoding="utf-8"), end="")
        else:
            target = Path(args.path)
            if target.exists():
                raise SystemExit(f"{target} already exists; not overwriting")
            target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {target}")
        return 0
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "grade":
        return _cmd_grade(args)
    if args.command == "make-caches":
        from hw2_dexhand.tools.make_caches import _main as make_caches_main

        return make_caches_main(args)
    overrides = list(args.overrides)
    if args.command == "play":
        overrides.append("training.play_only=true")
    from hw2_dexhand.eval.config import StudentConfigError

    try:
        cfg = _compose_target(args.task, args.config, overrides)
    except StudentConfigError as exc:
        raise SystemExit(f"hw2 {args.command}: {exc}") from None
    if args.cfg:
        print(OmegaConf.to_yaml(cfg))
        return 0
    _run_training(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
