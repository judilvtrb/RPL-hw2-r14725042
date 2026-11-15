"""Batched torch quaternion helpers, ``(w, x, y, z)`` convention.

A torch port of the NumPy helpers the original (NpEnv) reorientation env used
(``unilab.base.rotation``); ``tests/test_quaternion.py`` checks them against
those NumPy functions.
"""

from __future__ import annotations

import math

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions, shape ``(..., 4)``."""
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so that ``w >= 0`` (same rotation, unique representative)."""
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of unit quaternions (the conjugate)."""
    return quat_conjugate(quat_normalize(q))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b`` on ``(..., 4)`` tensors (broadcasting)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` ``(..., 3)`` by unit quaternions ``q`` ``(..., 4)``."""
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle in ``[0, pi]`` between unit quaternions ``a`` and ``b``."""
    dot = (quat_normalize(a) * quat_normalize(b)).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Rotation about world z by ``yaw`` radians, shape ``(..., 4)``."""
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def sample_uniform_quat(num: int, *, rng, device: torch.device) -> torch.Tensor:
    """``(num, 4)`` quaternions uniform on SO(3) (Shoemake's subgroup algorithm).

    ``rng`` is a UniLab ``ManagerRng`` (its ``uniform(low, high, shape)``).
    """
    u = rng.uniform(0.0, 1.0, (num, 3), dtype=torch.float32).to(device)
    u1, u2, u3 = u.unbind(-1)
    a = torch.sqrt(1.0 - u1)
    b = torch.sqrt(u1)
    two_pi = 2.0 * math.pi
    return torch.stack(
        [
            b * torch.cos(two_pi * u3),
            a * torch.sin(two_pi * u2),
            a * torch.cos(two_pi * u2),
            b * torch.sin(two_pi * u3),
        ],
        dim=-1,
    )


__all__ = [
    "quat_apply",
    "quat_canonicalize",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_inv",
    "quat_mul",
    "quat_normalize",
    "sample_uniform_quat",
    "yaw_to_quat",
]
