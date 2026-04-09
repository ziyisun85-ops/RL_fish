from __future__ import annotations

import math

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def heading_vector(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=float)


def world_to_body_vector(vector: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            cos_yaw * vector[0] + sin_yaw * vector[1],
            -sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=float,
    )


def body_to_world_vector(vector: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            cos_yaw * vector[0] - sin_yaw * vector[1],
            sin_yaw * vector[0] + cos_yaw * vector[1],
        ],
        dtype=float,
    )
