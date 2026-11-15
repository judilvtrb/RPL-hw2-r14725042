"""Generate grasp caches by simulation: ``hw2 make-caches [--objects ...]``.

For each object the collector builds the ``allegro_grasp/<object>`` owner
(keyframe pose + joint noise, zero actions, 3 s episodes, no auto-reset), and
at the end of every episode keeps the envs that still hold the object:

* every fingertip within ``max_fingertip_distance`` of the object,
* at least ``min_contacts`` fingertip contact sensors active,
* object above the owner's drop height (the episode was not terminated).

Kept states are ``[hand joints (16), object xyz (3), object wxyz (4)]`` in
float32, the layout ``reset_from_grasp_cache`` loads. Output goes to the
package cache directory (``${hw2_cache:...}``) unless ``--out-dir`` is given.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hw2_dexhand import CACHES_ROOT
from hw2_dexhand.cli import compose_config
from hw2_dexhand.eval.config import create_env
from hw2_dexhand.eval.tasks import OBJECTS

GRASP_OWNER = "allegro_grasp"
# Layout of the collection owner's ``grasp`` observation group.
_JOINTS = slice(0, 16)
_OBJECT_POS = slice(16, 19)
_OBJECT_QUAT = slice(19, 23)
_FINGERTIPS = slice(23, 35)
_CONTACTS = slice(35, 39)


def cache_name(obj: str, count: int) -> str:
    return (
        f"allegro_{obj}_grasp_{count // 1000}k.npy"
        if count % 1000 == 0
        else f"allegro_{obj}_grasp_{count}.npy"
    )


def collect_grasps(
    obj: str,
    *,
    count: int,
    num_envs: int,
    seed: int = 0,
    max_fingertip_distance: float = 0.1,
    min_contacts: int = 2,
    max_episodes: int = 1000,
    root_dir: str | Path = ".",
    log=print,
) -> np.ndarray:
    """Return ``(>= count, 23)`` stable grasps for ``obj`` (or fewer if ``max_episodes`` runs out)."""
    if obj not in OBJECTS:
        raise ValueError(f"object must be one of {OBJECTS}, got {obj!r}")
    cfg = compose_config("ppo", f"{GRASP_OWNER}/{obj}", ["training.device=cpu"])
    env = create_env(cfg, num_envs=int(num_envs), root_dir=root_dir)
    kept: list[np.ndarray] = []
    total = 0
    started = time.perf_counter()
    try:
        env.seed(int(seed))
        state = env.init_state()
        zeros = torch.zeros((env.num_envs, env.action_space.shape[0]), dtype=torch.float32)
        episodes = 0
        while total < count and episodes < max_episodes:
            state = env.step(zeros)
            done = state.terminated | state.truncated
            if not bool(done.any()):
                continue
            # auto_reset is off: the final state of finished envs is still visible.
            survived = state.truncated & ~state.terminated
            if bool(survived.any()):
                grasp = state.obs["critic"]
                tips = grasp[:, _FINGERTIPS].reshape(env.num_envs, -1, 3)
                obj_pos = grasp[:, _OBJECT_POS]
                near = (
                    torch.linalg.norm(tips - obj_pos.unsqueeze(1), dim=-1) < max_fingertip_distance
                ).all(dim=1)
                contacts = (grasp[:, _CONTACTS] > 0.5).sum(dim=1) >= int(min_contacts)
                good = survived & near & contacts
                if bool(good.any()):
                    rows = grasp[good][:, :23].detach().cpu().numpy().astype(np.float32)
                    kept.append(rows)
                    total += int(rows.shape[0])
            episodes += 1
            env.reset(torch.nonzero(done).flatten())
            if episodes % 5 == 0 or total >= count:
                log(
                    f"[{obj}] episode {episodes}: {total}/{count} grasps "
                    f"({time.perf_counter() - started:.0f}s)"
                )
    finally:
        env.close()
    if not kept:
        return np.zeros((0, 23), dtype=np.float32)
    return np.concatenate(kept, axis=0)[:count]


def make_caches(
    objects: list[str],
    *,
    count: int = 10000,
    num_envs: int = 1024,
    seed: int = 0,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
    root_dir: str | Path = ".",
    log=print,
) -> dict[str, Path]:
    out = Path(out_dir) if out_dir is not None else CACHES_ROOT
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for obj in objects:
        path = out / cache_name(obj, count)
        if path.is_file() and not overwrite:
            log(f"[{obj}] {path.name} exists; skipping (use --overwrite to rebuild)")
            written[obj] = path
            continue
        grasps = collect_grasps(
            obj, count=count, num_envs=num_envs, seed=seed, root_dir=root_dir, log=log
        )
        if grasps.shape[0] == 0:
            raise RuntimeError(f"[{obj}] no stable grasps collected")
        np.save(path, grasps)
        log(f"[{obj}] wrote {grasps.shape[0]} grasps to {path}")
        written[obj] = path
    return written


def _main(args: Any) -> int:
    objects = list(args.objects) if args.objects else list(OBJECTS)
    make_caches(
        objects,
        count=int(args.count),
        num_envs=int(args.num_envs),
        seed=int(args.seed),
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
        root_dir=Path.cwd(),
    )
    return 0


__all__ = ["GRASP_OWNER", "cache_name", "collect_grasps", "make_caches"]
