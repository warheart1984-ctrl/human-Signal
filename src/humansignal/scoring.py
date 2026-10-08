"""Shared numeric helpers. Scores are deterministic squashes, not probabilities."""

from __future__ import annotations

import math


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def unit(value: float) -> float:
    """Clamp to [0, 1] and round to 4 decimals for stable JSON."""
    return round(clamp(value), 4)


def squash(raw: float) -> float:
    """Map a non-negative raw mass to [0, 1) with diminishing returns.

    ``0 -> 0``, ``1 -> ~0.63``, ``2 -> ~0.86``. Negative input is treated as 0.
    """
    if raw <= 0.0:
        return 0.0
    return 1.0 - math.exp(-raw)
