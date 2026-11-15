"""HW2 dexterous-hand task package for UniLab.

Importing this package registers the ``hw2_assets`` OmegaConf resolver so that
owner YAMLs can reference packaged assets without absolute paths::

    model_file: ${hw2_assets:robots/allegro_hand/scene_ball.xml}

UniLab imports ``hw2_dexhand.tasks`` through the ``unilab.tasks`` entry point,
which imports this module first, so the resolver is also available inside
spawned collector processes.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
CONF_ROOT = PACKAGE_ROOT / "conf"

CACHES_ROOT = ASSETS_ROOT / "caches"
TEMPLATES_ROOT = PACKAGE_ROOT / "templates"

ASSET_RESOLVER_NAME = "hw2_assets"
CACHE_RESOLVER_NAME = "hw2_cache"


def asset_path(relative: str) -> str:
    """Return the absolute path of a packaged asset, failing if it is missing."""
    path = ASSETS_ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"HW2 asset not found: {path}")
    return str(path)


def cache_path(name: str) -> str:
    """Return the path of a generated grasp cache without requiring it to exist.

    Caches are generated locally (``hw2 make-caches``) and are not shipped, so
    a missing file is reported by the term that loads it, with the remedy.
    """
    return str(CACHES_ROOT / name)


if not OmegaConf.has_resolver(ASSET_RESOLVER_NAME):
    OmegaConf.register_new_resolver(ASSET_RESOLVER_NAME, asset_path)
if not OmegaConf.has_resolver(CACHE_RESOLVER_NAME):
    OmegaConf.register_new_resolver(CACHE_RESOLVER_NAME, cache_path)

__all__ = [
    "ASSETS_ROOT",
    "CACHES_ROOT",
    "CONF_ROOT",
    "PACKAGE_ROOT",
    "TEMPLATES_ROOT",
    "asset_path",
    "cache_path",
]
