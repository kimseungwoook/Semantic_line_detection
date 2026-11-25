import numpy as np
import math

def nms_2d_topk(heatmap, K=6, radius=2, threshold=0.2):
    """
    예측 Hough Heatmap에서 높은 점수의 Peak를 순차 선택하고,
    주변 후보를 억제하여 중복 직선 검출을 줄이는 함수

    heatmap: sigmoid를 적용한 2차원 NumPy 배열 [R, Theta]
    K: 선택할 Peak의 최대 개수
    radius: 선택한 Peak 주변의 억제 반경. (ex:radius=2이면 중심을 포함한 5×5 영역을 억제)
    threshold: Peak를 선택할 최소 점수. 이 값 이상이면 선택 가능

    return:
    점수 내림차순의 (rho_idx, theta_idx, score) 목록.
    조건을 만족하는 후보가 없으면 빈 리스트를 반환한다.
    """
    hmap = heatmap.copy()
    peaks = []

    for _ in range(K):
        idx = np.unravel_index(np.argmax(hmap), hmap.shape)
        peak_val = hmap[idx]

        if peak_val < threshold:
            break

        peaks.append((idx[0], idx[1], peak_val))

        y, x =  idx
        y0 = max(0, y - radius)
        y1 = min(hmap.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(hmap.shape[1], x + radius + 1)
        hmap[y0:y1, x0:x1] = 0.0

    return peaks

def hough_bin_to_line_points(peak, original_w=346, original_h=260, input_w=224, input_h=224, R=112, Theta=100):
    # 허프 공간 점을 원본 이미지의 직선으로 복원하는 함수

    # 허프 공간 peak
    rho_idx, theta_idx = peak[:2]

    # 입력 이미지 중심
    center_x = (input_w - 1) / 2
    center_y = (input_h - 1) / 2

    rho_max = math.sqrt(center_x ** 2 + center_y ** 2)
    rho_step = 2 * rho_max / R

    # hough space의 rho_idx, theta_idx ---> 입력 이미지의 rho, theta로 변환
    rho = -rho_max + (rho_idx + 0.5) * rho_step
    theta = theta_idx * math.pi / Theta

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    candidates = []

    # rho = (x-center_x) * cos_theta + (y-center_y) * sin_theta
    # semantic_line은 좌측 평면, 우측 평면, 위 평면, 아래 평면에서 해당 식이 구해지는 것만 구하면 됨

    # 1. 좌측 평면 : x=0, y=0~223
    if abs(sin_theta) > 1e-8:
        x = 0
        y = center_y + (rho + center_x * cos_theta) / sin_theta

        if 0 <= y <= input_h-1:
            candidates.append((x, y))

    # 2. 우측 평면 : x=223, y=0~223
    if abs(sin_theta) > 1e-8:
        x = input_w - 1
        y = center_y + (rho-(x-center_x)*cos_theta) / sin_theta

        if 0 <= y <= input_h-1:
            candidates.append((x, y))

    # 3. 위 평면 : x=0~223, y=0
    if abs(cos_theta) > 1e-8:
        x = center_x + (rho + center_y * sin_theta) / cos_theta
        y = 0

        if 0 <= x <= input_w - 1:
            candidates.append((x, y))

    # 4. 아래 평면 : x=0~223, y=223
    if abs(cos_theta) > 1e-8:
        y = input_h - 1
        x = center_x + (rho-(y-center_y)*sin_theta) / cos_theta

        if 0 <= x <= input_w - 1:
            candidates.append((x, y))

    points = []

    # 중복 좌표 제거
    for x, y in candidates:
        duplicate = False

        for px, py in points:
            if  math.hypot(x - px, y - py) < 1e-8:
                duplicate=True
                break
        if not duplicate:
            points.append((x, y))

    if len(points) < 2:
        return None

    original_x = []
    original_y = []

    for x, y in points:
        ori_x = x * (original_w / input_w)
        ori_y = y * (original_h / input_h)

        original_x.append(ori_x)
        original_y.append(ori_y)

    return ((original_x[0], original_y[0]), (original_x[-1], original_y[-1]))

