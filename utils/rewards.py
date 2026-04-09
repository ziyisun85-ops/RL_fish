from __future__ import annotations

import math


def compute_obstacle_penalty(clearance: float, safety_margin: float) -> float:
    if not math.isfinite(clearance):
        return 0.0
    if clearance >= safety_margin:
        return 0.0
    return max(0.0, 1.0 - clearance / max(safety_margin, 1e-6))
