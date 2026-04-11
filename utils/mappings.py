from __future__ import annotations

import math

import numpy as np

from configs.default_config import MappingConfig


def servo_angle_to_head_angle(theta_m: float, mapping: MappingConfig) -> float:
    theta_m_limit = max(abs(float(mapping.theta_m_max)), 1e-8)
    normalized_theta_m = float(np.clip(theta_m / theta_m_limit, -1.0, 1.0))
    return float(math.radians(mapping.head_angle_max_deg) * normalized_theta_m)


def head_angle_to_tail_frequency(theta_h: float, mapping: MappingConfig) -> float:
    modulation = 1.0 + mapping.tail_freq_gain * abs(theta_h)
    frequency = mapping.base_tail_freq * modulation
    return float(np.clip(frequency, mapping.tail_freq_min, mapping.tail_freq_max))


def head_angle_to_phase_rate(theta_h: float, mapping: MappingConfig) -> float:
    return float(2.0 * math.pi * head_angle_to_tail_frequency(theta_h, mapping))
