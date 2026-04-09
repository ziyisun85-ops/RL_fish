from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from configs.default_config import PathConfig


@dataclass
class PathQuery:
    closest_point: np.ndarray
    tangent: np.ndarray
    heading: float
    cross_track_error: float
    arc_length: float
    progress_ratio: float


def generate_reference_path(path_config: PathConfig) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.linspace(path_config.start_x, path_config.end_x, path_config.num_points, dtype=float)
    if path_config.kind == "straight":
        y_values = np.zeros_like(x_values)
    elif path_config.kind == "sine":
        normalized = (x_values - path_config.start_x) / max(path_config.end_x - path_config.start_x, 1e-6)
        y_values = path_config.amplitude * np.sin(
            2.0 * math.pi * normalized * (path_config.end_x - path_config.start_x) / path_config.wavelength
        )
    else:
        raise ValueError(f"Unsupported path kind: {path_config.kind}")

    points = np.column_stack([x_values, y_values])
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return points, cumulative_lengths


def project_point_to_path(position: np.ndarray, path_points: np.ndarray, cumulative_lengths: np.ndarray) -> PathQuery:
    best_distance_sq = math.inf
    best_projection = path_points[0]
    best_tangent = np.array([1.0, 0.0], dtype=float)
    best_cross_track = 0.0
    best_arc_length = 0.0

    for index in range(len(path_points) - 1):
        start = path_points[index]
        end = path_points[index + 1]
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-10:
            continue
        projection_alpha = float(np.clip(np.dot(position - start, segment) / segment_length_sq, 0.0, 1.0))
        projection = start + projection_alpha * segment
        offset = position - projection
        distance_sq = float(np.dot(offset, offset))
        if distance_sq >= best_distance_sq:
            continue

        tangent = segment / math.sqrt(segment_length_sq)
        cross_track = tangent[0] * offset[1] - tangent[1] * offset[0]
        best_distance_sq = distance_sq
        best_projection = projection
        best_tangent = tangent
        best_cross_track = cross_track
        best_arc_length = float(cumulative_lengths[index] + projection_alpha * math.sqrt(segment_length_sq))

    heading = float(math.atan2(best_tangent[1], best_tangent[0]))
    progress_ratio = best_arc_length / max(float(cumulative_lengths[-1]), 1e-6)
    return PathQuery(
        closest_point=best_projection.astype(float),
        tangent=best_tangent.astype(float),
        heading=heading,
        cross_track_error=float(best_cross_track),
        arc_length=float(best_arc_length),
        progress_ratio=float(np.clip(progress_ratio, 0.0, 1.0)),
    )


def _interpolate_along_path(path_points: np.ndarray, cumulative_lengths: np.ndarray, arc_length: float) -> np.ndarray:
    if arc_length <= 0.0:
        return path_points[0].copy()
    if arc_length >= cumulative_lengths[-1]:
        return path_points[-1].copy()

    upper_index = int(np.searchsorted(cumulative_lengths, arc_length, side="right"))
    lower_index = max(0, upper_index - 1)
    start_length = cumulative_lengths[lower_index]
    end_length = cumulative_lengths[upper_index]
    alpha = (arc_length - start_length) / max(end_length - start_length, 1e-8)
    return path_points[lower_index] + alpha * (path_points[upper_index] - path_points[lower_index])


def get_lookahead_target(
    query: PathQuery,
    path_points: np.ndarray,
    cumulative_lengths: np.ndarray,
    lookahead_distance: float,
) -> np.ndarray:
    target_arc_length = min(query.arc_length + lookahead_distance, float(cumulative_lengths[-1]))
    return _interpolate_along_path(path_points, cumulative_lengths, target_arc_length)
