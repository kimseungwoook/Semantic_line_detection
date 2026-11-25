import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def EA_score(pred_line, gt_line, image_size=(260, 346)):
    (pred_x1, pred_y1), (pred_x2, pred_y2) = pred_line
    (gt_x1, gt_y1), (gt_x2, gt_y2) = gt_line

    # 각도 점수
    pred_angle = math.atan2(pred_y2 - pred_y1, pred_x2 - pred_x1) % math.pi
    gt_angle = math.atan2(gt_y2 - gt_y1, gt_x2 - gt_x1) % math.pi

    angle_difference = abs(pred_angle-gt_angle)
    angle_difference = min(
        angle_difference,
        math.pi - angle_difference
    )

    angle_score = (1 - angle_difference / (math.pi / 2)) ** 2

    # 위치 점수
    # 예측 직선의 중점
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2

    # GT 직선의 중점
    gt_center_x = (gt_x1 + gt_x2) / 2
    gt_center_y = (gt_y1 + gt_y2) / 2

    # 두 중점 사이 거리
    distance = math.hypot(
        pred_center_x - gt_center_x,
        pred_center_y - gt_center_y
    )

    # 이미지의 긴 변으로 정규화
    normalized_distance = distance / max(image_size)
    position_score = max(0.0, 1 - normalized_distance) ** 2

    ea_score = angle_score * position_score

    return ea_score

def match_lines(pred_lines, gt_lines, ea_threshold, image_size=(260, 346)):
    pred_lines = [
        line for line in pred_lines
        if math.dist(line[0], line[1]) > 0
    ]
    gt_lines = [
        line for line in gt_lines
        if math.dist(line[0], line[1]) > 0
    ]

    pred_count = len(pred_lines)
    gt_count = len(gt_lines)

    # 예측이나 GT가 없는 경우
    if pred_count == 0 or gt_count == 0:
        return 0, pred_count, gt_count

    # 행은 예측 직선, 열은 GT 직선
    graph = np.zeros((pred_count, gt_count))

    for pred_idx, pred_line in enumerate(pred_lines):
        for gt_idx, gt_line in enumerate(gt_lines):
            score = EA_score(
                pred_line,
                gt_line,
                image_size=image_size
            )

            if score >= ea_threshold:
                graph[pred_idx, gt_idx] = 1

    # 정답으로 인정되는 연결 개수를 최대화
    pred_indices, gt_indices = linear_sum_assignment(-graph)

    tp = int(graph[pred_indices, gt_indices].sum())
    fp = pred_count - tp
    fn = gt_count - tp

    return tp, fp, fn

def precision_recall_f1(tp, fp, fn):
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1

import numpy as np


def get_region_masks(lines, image_size=(260, 346)):
    """
    직선들이 이미지를 나누는 영역별 마스크를 반환한다.

    lines:
        [
            ((x1, y1), (x2, y2)),
            ...
        ]

    image_size: (높이, 너비)
    반환: 영역마다 [높이, 너비] 크기의 bool 배열
    """
    height, width = image_size
    y_grid, x_grid = np.indices((height, width))

    # 처음에는 전체 이미지가 영역 하나
    region_ids = np.zeros((height, width), dtype=np.int64)

    for line in lines:
        (x1, y1), (x2, y2) = line

        # 길이가 0인 선은 제외
        if x1 == x2 and y1 == y2:
            continue

        # 각 픽셀이 직선의 어느 쪽에 있는지 계산
        value = (
            (y2 - y1) * x_grid
            - (x2 - x1) * y_grid
            + x2 * y1
            - y2 * x1
        )
        side = (value > 0).astype(np.int64)

        # 기존 영역을 현재 직선의 양쪽으로 분리
        region_ids = region_ids * 2 + side

        # 영역 번호를 0부터 다시 부여
        _, inverse = np.unique(region_ids, return_inverse=True)
        region_ids = inverse.reshape(height, width)

    masks = []

    for region_id in np.unique(region_ids):
        masks.append(region_ids == region_id)

    return masks


def calculate_hiou(pred_lines, gt_lines, image_size=(260, 346)):
    """
    예측 직선과 GT 직선이 만드는 영역 분할의 HIoU를 계산한다.

    pred_lines, gt_lines:
        [
            ((x1, y1), (x2, y2)),
            ...
        ]

    image_size: (높이, 너비)
    반환: 0~1 범위의 HIoU
    """
    pred_masks = get_region_masks(pred_lines, image_size)
    gt_masks = get_region_masks(gt_lines, image_size)

    pred_count = len(pred_masks)
    gt_count = len(gt_masks)

    iou_matrix = np.zeros((pred_count, gt_count), dtype=np.float64)

    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            intersection = np.count_nonzero(pred_mask & gt_mask)
            union = np.count_nonzero(pred_mask | gt_mask)

            if union > 0:
                iou_matrix[pred_idx, gt_idx] = intersection / union

    # 각 예측 영역과 가장 잘 겹치는 GT 영역
    pred_to_gt = iou_matrix.max(axis=1).sum()

    # 각 GT 영역과 가장 잘 겹치는 예측 영역
    gt_to_pred = iou_matrix.max(axis=0).sum()

    hiou = (pred_to_gt + gt_to_pred) / (pred_count + gt_count)

    return float(hiou)