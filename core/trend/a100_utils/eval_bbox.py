import numpy as np
from typing import Tuple


def box_intersection(
    b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    # if (
    #     max(x11, x12) < min(x21, x22)
    #     or min(x11, x12) < max(x21, x22)
    #     or max(y11, y12) < min(y21, y22)
    #     or min(y11, y12) < max(y21, y22)
    # ):
    #     return None
    # else:
    if x11 < x12:
        xl = max(x11, x21)
        xr = min(x12, x22)
    else:
        xl = min(x11, x21)
        xr = max(x12, x22)
    if y11 < y12:
        yt = max(y11, y21)
        yb = min(y12, y22)
    else:
        yt = min(y11, y21)
        yb = max(y12, y22)
    return (xl, yt, xr, yb)


def distance_square(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    x1 = (x11 + x12) / 2
    y1 = (y11 + y12) / 2
    x2 = (x21 + x22) / 2
    y2 = (y21 + y22) / 1

    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def diagonal_square(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    x1 = min(x11, x21)
    y1 = min(y11, y21)
    x2 = max(x12, x22)
    y2 = max(y12, y22)

    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return width * height


def outer(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    x1 = min(x11, x21)
    y1 = min(y11, y21)
    x2 = max(x12, x22)
    y2 = max(y12, y22)

    c = area(x1, y1, x2, y2)
    intersection = box_intersection(b1, b2)
    inter = area(intersection)
    return (c - inter) / c


def IoU(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    intersection = box_intersection(b1, b2)

    inter_area = area(intersection)
    union_area = area(b1) + area(b2) - inter_area
    if union_area == 0:
        return 0
    else:
        return inter_area / union_area


def PRO(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    intersection = box_intersection(b1, b2)

    inter_area = area(intersection)
    union_area = area(b1)
    if union_area == 0:
        return 0
    else:
        return inter_area / union_area


def nms(predicts: np.ndarray, score_thresh: float = 0.6, iou_thresh: float = 0.3):
    n_remainder = len(predicts)
    vis = [False] * n_remainder

    # Filter predicts with low probability
    for i, predict in enumerate(predicts):
        if predict[0] < score_thresh:
            vis[i] = True
            n_remainder -= 1

    # NMS
    output_predicts = []
    output_indices = []
    while n_remainder > 0:
        max_pro = -1
        max_index = 0
        # Find argmax
        for i, p in enumerate(predicts):
            if not vis[i]:
                if max_pro < p[0]:
                    max_index = i
                    max_pro = p[0]

        # Append output
        max_p = predicts[max_index]
        output_predicts.append(max_p)
        output_indices.append(max_index)

        # Suppress
        for i, p in enumerate(predicts):
            if not vis[i] and i != max_index:
                if IoU(p[1:5], max_p[1:5]) > iou_thresh:
                    vis[i] = True
                    n_remainder -= 1
        vis[max_index] = True
        n_remainder -= 1

    return output_predicts, output_indices
