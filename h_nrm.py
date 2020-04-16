"""Contains methods which help to split shots into scenes using the H_nrm cost function."""

from typing import Set

import numpy as np
from memorize import memorize


def distance_sum(distance_matrix: np.ndarray, j1: int, j2: int) -> float:
    """
    Get sum of distances in square of size j2-j1 located in [j1;j2] position.

    :param distance_matrix: pairwise distances matrix
    :param j1: first index
    :param j2: second index
    :return: sum of distances
    """
    return np.sum(distance_matrix[(j1 - 1):j2, (j1 - 1):j2])


@memorize()
def get_embedded_areas_sums(parent_square_size: int, embedded_squares_count: int) -> Set[int]:
    """
    Get the set of all possible areas which can found by dividing a square of the sizeparent_square_size
    into embedded_squares_count subsquares. The function works in the recursive manner.

    :param parent_square_size: size of the parent square
    :param embedded_squares_count: count of parts to divide
    :return: set of possible areas
    """
    if embedded_squares_count == 0:
        return {0}
    if embedded_squares_count == 1:
        return {parent_square_size ** 2}
    if parent_square_size / embedded_squares_count < 1:
        return set()

    sums = set()
    for i in range(int(round(parent_square_size / 2))):
        area = (i + 1) ** 2
        for ps in get_embedded_areas_sums(parent_square_size - i - 1, embedded_squares_count - 1):
            sums.add(area + ps)
    return sums


def get_optimal_sequence_nrm(distance_matrix: np.ndarray, scenes_count: int) -> np.ndarray:
    """
   Divide shots into scenes using the H_nrm cost function.

    More info in article: https://www.ibm.com/blogs/research/2018/09/video-scene-detection/

    :param distance_matrix: matrix of pairwise distances between shots
    :param scenes_count: number of resulting scenes
    :return: indexes of the last shot of each scene
    """
    D = distance_matrix
    K = scenes_count
    N = D.shape[0]
    C, J, P = {}, {}, {}

    k = 1
    for n in range(1, N + 1):
        _area = (N - n + 1) ** 2
        for p in get_embedded_areas_sums(n - 1, K - k):
            _dist = distance_sum(D, n, N)
            J[(n, k, p)] = N
            P[(n, k, p)] = _area
            C[(n, k, p)] = _dist / (p + _area)

    for k in range(2, K + 1):
        for n in range(1, N):
            if (N - n + 1) < k:
                continue

            for p in get_embedded_areas_sums(n - 1, K - k):
                min_C = np.inf
                min_i = np.iinfo(np.int32).max
                for i in range(n, N):
                    if (N - i) < k - 1:
                        continue

                    cur_area = (i - n + 1) ** 2
                    next_area = P.get((i + 1, k - 1, p + cur_area), 0)
                    G = distance_sum(D, n, i) / (p + cur_area + next_area)
                    c = G + C.get((i + 1, k - 1, p + cur_area), 0)

                    if c < min_C:
                        min_C = c
                        min_i = i

                C[(n, k, p)] = min_C
                J[(n, k, p)] = min_i
                P[(n, k, p)] = (min_i - n + 1) ** 2 + P.get((min_i + 1, k - 1, p + (min_i - n + 1) ** 2), 0)

    t = [0]
    P_tot = 0
    for i in range(1, K + 1):
        t.append(J[(t[-1] + 1, K - i + 1, P_tot)])
        P_tot += (t[-1] - t[-2]) ** 2
    return np.array(t[1:]) - 1
