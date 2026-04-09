from __future__ import annotations

import math

import numpy as np

from configs.default_config import MappingConfig


def servo_angle_to_head_angle(theta_m: float, mapping: MappingConfig) -> float:
    transmission_denominator = max(mapping.head_transmission_efficiency * mapping.head_joint_arm_length, 1e-8)
    head_argument = mapping.servo_output_radius * theta_m / transmission_denominator
    return float(np.arcsin(np.clip(head_argument, -1.0, 1.0)))


def head_angle_to_tail_frequency(theta_h: float, mapping: MappingConfig) -> float:
    modulation = 1.0 + mapping.tail_freq_gain * float(np.sign(theta_h)) * abs(theta_h)
    frequency = mapping.base_tail_freq * modulation
    return float(np.clip(frequency, mapping.tail_freq_min, mapping.tail_freq_max))


def head_angle_to_phase_rate(theta_h: float, mapping: MappingConfig) -> float:
    return float(2.0 * math.pi * head_angle_to_tail_frequency(theta_h, mapping))
