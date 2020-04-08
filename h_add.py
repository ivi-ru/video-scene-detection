"""Contains methods which help to split shots into scenes using the H_add cost function."""

import numpy as np


def get_optimal_sequence_add(distance_matrix: np.ndarray, scenes_count: int) -> np.ndarray:
    """
    Divide shots into scenes regarding to H_add metrics.

    More info in paper: https://ieeexplore.ieee.org/abstract/document/7823628

    :param distance_matrix: matrix of pairwise distances between shots
    :param scenes_count: number of resulting scenes
    :return: indexes of the last shot of each scene
    """
    D = distance_matrix
    K = scenes_count
    N = len(D)
    C = np.zeros((N, K))
    J = np.zeros((N, K), dtype=int)

    for n in range(0, N):
        C[n, 0] = np.sum(D[n:, n:])

    for n in range(0, N):
        J[n, 0] = N - 1

    for k in range(1, K):
        for n in range(0, N):
            candidates = []
            for i in range(n, N):
                if i < N - 1:
                    C_prev = C[i + 1, k - 1]
                else:
                    C_prev = 0
                h_n_i = np.sum(D[n:i + 1, n:i + 1])
                candidate = h_n_i + C_prev
                candidates.append(candidate)
            candidates = np.array(candidates)
            C[n, k] = np.min(candidates)
            J[n, k] = np.where(candidates == C[n, k])[0][0] + n

    t = np.zeros((K,), dtype=int)
    t_prev = 0
    for i in range(0, K):
        if i == 0:
            t_prev = 0
        else:
            t_prev = t[i - 1]
        t[i] = J[t_prev, K - i - 1]
    return t
