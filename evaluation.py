"""Contains functions for the evaluation of the scene detection results."""

from typing import List, Tuple


def intersection(
        gt_begin: float,
        gt_end: float,
        pred_begin: float,
        pred_end: float
) -> float:
    """
    Get intersection for two intervals.

    :param gt_begin: truth start point
    :param gt_end: truth end point
    :param pred_begin: predicted start point
    :param pred_end: predicted end point
    :return: intersection width
    """
    return max(min(gt_end, pred_end) - max(gt_begin, pred_begin), 0)

def intersection_over_union(
        gt_begin: float,
        gt_end: float,
        pred_begin: float,
        pred_end: float
) -> float:
    """
    Get IoU for two intervals.

    :param gt_begin: truth start point
    :param gt_end: truth end point
    :param pred_begin: predicted start point
    :param pred_end: predicted end point
    :return: IoU
    """
    int_len = intersection(gt_begin, gt_end, pred_begin, pred_end)
    pred_len = pred_end - pred_begin
    gt_len = gt_end - gt_begin

    return int_len / (pred_len + gt_len - int_len + 1e-32)

def precision(
        gt_begin: float,
        gt_end: float,
        pred_begin: float,
        pred_end: float
) -> float:
    """
    Get precision by dividing the intersection length of the predicted
    and the ground truth intervals to the length of the predicted interval.

    :param gt_begin: truth start point
    :param gt_end: truth end point
    :param pred_begin: predicted start point
    :param pred_end: predicted end point
    :return: значение precision
    """
    int_len = intersection(gt_begin, gt_end, pred_begin, pred_end)
    pred_len = pred_end - pred_begin

    return int_len / (pred_len + 1e-32)


def recall(
        gt_begin: float,
        gt_end: float,
        pred_begin: float,
        pred_end: float
) -> float:
    """
    Get recall by dividing the intersection length of the predicted
    and the ground truth intervals to the length of the ground truth interval.

    :param gt_begin: truth start point
    :param gt_end: truth end point
    :param pred_begin: predicted start point
    :param pred_end: predicted end point
    :return: значение recall
    """
    int_len = intersection(gt_begin, gt_end, pred_begin, pred_end)
    gt_len = gt_end - gt_begin

    return int_len / (gt_len + 1e-32)


def f1(
        gt_begin: float,
        gt_end: float,
        pred_begin: float,
        pred_end: float
) -> float:
    """
    Get f1-score for two intervals.

    :param gt_begin: truth start point
    :param gt_end: truth end point
    :param pred_begin: predicted start point
    :param pred_end: predicted end point
    :return: f1-score value
    """
    p = precision(gt_begin, gt_end, pred_begin, pred_end)
    r = recall(gt_begin, gt_end, pred_begin, pred_end)
    return 2 * p * r / (p + r + 1e-32)


def calculate_interval_metric(
        gt_data: List[List[Tuple[int, int]]],
        pred_data: List[List[Tuple[int, int]]],
        metric_name: str
) -> float:
    """
    Get the average value of the metric_name metric for the predicted intervals.

    :param gt_data: ground truth intervals
    :param pred_data: predicted intervals
    :param metric_name: metric name ('precision', 'recall', 'f1')
    :return: aggregated value of metric
    """
    result = []
    for i, gt_intervals in enumerate(gt_data):
        for gt_begin, gt_end in gt_intervals:
            metric_values = []
            for pred_begin, pred_end in pred_data[i]:
                if metric_name == 'precision':
                    metric_value = precision(gt_begin, gt_end, pred_begin, pred_end)
                elif metric_name == 'recall':
                    metric_value = recall(gt_begin, gt_end, pred_begin, pred_end)
                elif metric_name == 'f1':
                    metric_value = f1(gt_begin, gt_end, pred_begin, pred_end)
                else:
                    metric_value = intersection_over_union(
                        gt_begin,
                        gt_end,
                        pred_begin,
                        pred_end
                    )
                metric_values.append(metric_value)

            best_value = max(metric_values)

            result.append(best_value)

    return sum(result) / len(result) if result else 0
