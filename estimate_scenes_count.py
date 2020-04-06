"""Contains method which estimates scenes count for video scene detection."""

import numpy as np
from numpy.linalg import norm


def estimate_scenes_count(distance_matrix: np.ndarray) -> int:
    """
    Calculate approximate count of scenes.

    Get singular values of the distance_matrix and then - index of the "elbow value".

    :paran distance_matrix: matrix of the pairvaise distances between shots
    :return: estimated count of scenes
    """
    singular_values = np.linalg.svd(distance_matrix, full_matrices=False, compute_uv=False)
    singular_values = singular_values[:len(singular_values) // 2]
    singular_values = np.log(singular_values)

    start_point = np.array([0, singular_values[0]])
    end_point = np.array([len(singular_values), singular_values[-1]])
    max_distance = 0
    elbow_point = 0
    for i, singular_value in enumerate(singular_values):
        current_point = np.array([i, singular_value])
        distance = norm(np.cross(start_point - end_point, start_point - current_point)) / \
            norm(end_point - start_point)
        if distance > max_distance:
            max_distance = distance
            elbow_point = i
    return elbow_point
