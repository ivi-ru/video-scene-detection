from typing import Iterator, Set

import numpy as np
from memorize import memorize


def distance_sum(distance_matrix: np.ndarray, j1: int, j2: int):
    """
    Gets sum of distances in square of size j2-j1 located in [j1;j2] position

    :param distance_matrix: pairwise distances matrix
    :param j1: first index
    :param j2: second index
    :return: sum of distances
    """
    return np.sum(distance_matrix[(j1 - 1):j2, (j1 - 1):j2])


@memorize()
def get_embedded_areas_sums(parent_square_size: int, embedded_squares_count: int) -> Set[int]:
    """
    Gets all possible areas we can get if we divide square with side of parent_square_size
    on embedded_squares_count parts in recursive manner

    :param parent_square_size: size of the parent square
    :param embedded_squares_count: count of parts to divide
    :return: set of possible areas:
    """
    if embedded_squares_count == 0:
        return {0}
    if embedded_squares_count == 1:
        return {parent_square_size ** 2}
    if parent_square_size / embedded_squares_count < 1:
        return {}

    sums = set()
    for i in range(int(round(parent_square_size / 2))):
        area = (i + 1) ** 2
        for ps in get_embedded_areas_sums(parent_square_size - i - 1, embedded_squares_count - 1):
            sums.add(area + ps)
    return sums


def get_embedded_areas_sums_non_recursive(parent_square_size: int, embedded_squares_count: int) -> Iterator[int]:
    """
    Gets all possible areas we can get if we divide square with side of parent_square_size
    on embedded_squares_count parts

    :param parent_square_size: size of the parent square
    :param embedded_squares_count: count of parts to divide
    :return: iterator of possible areas:
    """
    indexer = [0] * (embedded_squares_count - 1)

    while True:
        counter = int((parent_square_size - sum((k + 1) for k in indexer[:-1])) / 2)

        for j in range(indexer[-1], counter):
            indexer[-1] = j
            used_area = sum((k + 1) ** 2 for k in indexer)
            used_size = sum((k + 1) for k in indexer)
            total_area = used_area + (parent_square_size - used_size) ** 2
            yield total_area

        for pointer in range(len(indexer) - 1, -1, -1):
            max_iter = int(
                (parent_square_size - sum((k + 1) for k in indexer[:pointer])) / (embedded_squares_count - pointer)
            )
            if indexer[pointer] < max_iter - 1:
                indexer[pointer] += 1
                for k in range(pointer + 1, len(indexer)):
                    indexer[k] = indexer[pointer]
                break
        else:
            break


def get_optimal_sequence_nrm(distance_matrix: np.ndarray, scenes_count: int) -> np.ndarray:
    """
    Calculates dividing shots into scenes regarding to H_nrm metrics

    More info in article: https://www.ibm.com/blogs/research/2018/09/video-scene-detection/

    :param distance_matrix: matrix pf pairwise distances between shots
    :param scenes_count: number of scenes you want to divide your shots
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
                min_i = None
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
    assert P_tot == P[1, K, 0]
    return np.array(t[1:]) - 1
