# Annotation의 직선들을 정답 Hough heatmap으로 생성
import math
import torch

def resize_line_points(points, original_w=346, original_h=260, target_w=224, target_h=224):
    resized_points = []
    for point in points:
        x = point[0] * target_w / original_w
        y = point[1] * target_h / original_h

        resized_points.append([x, y])

    return resized_points

def line_to_hough(points, image_w=224, image_h=224):
    # 직선의 두 끝 점을 Hough 파라미터로 변환
    center_x = (image_w-1) / 2
    center_y = (image_h-1) / 2

    x0 = float(points[0][0]) - center_x
    y0 = float(points[0][1]) - center_y
    x1 = float(points[1][0]) - center_x
    y1 = float(points[1][1]) - center_y

    # 방향 벡터
    dx = x1 - x0
    dy = y1 - y0

    # 직선에 수직인 벡터
    normal_x = dy
    normal_y = -dx

    if dx == 0 and dy == 0:
        raise ValueError("두 끝점이 같아서 직선을 정의할 수 없습니다.")

    # 법선 각도를 0 이상 π 미만으로 맞춘다.
    theta = math.atan2(normal_y, normal_x) % math.pi

    # 각도를 정규화한 뒤 해당 방향의 rho를 계산한다.
    rho = x0 * math.cos(theta) + y0 * math.sin(theta)

    return rho, theta

def hough_to_bin(rho, theta, image_w=224, image_h=224, R=112, THETA=100):
    # Hough 파라미터를 heatmap의 행열로 변환
    center_x = (image_w - 1) / 2
    center_y = (image_h - 1) / 2

    rho_max = math.sqrt(center_x**2 + center_y**2)
    rho_step = rho_max * 2 / R

    theta_step = math.pi / THETA
    # 실제 각도에서 가장 가까운 theta bin 선택
    theta_idx = math.floor(theta / theta_step + 0.5)

    # 180°로 반올림되면 0°로 옮기고 rho 부호를 반전한다.
    if theta_idx == THETA:
        theta_idx = 0
        rho = -rho

    rho_idx = math.floor((rho_max + rho) / rho_step)
    rho_idx = max(0, min(R - 1, rho_idx))

    return rho_idx, theta_idx

def make_line_heatmap(
    rho_idx,
    theta_idx,
    R,
    THETA,
    sigma_rho=1.0,
    sigma_theta=1.0,
    mode="gaussian",
    kernel_size=5,
):
    """
    직선 하나의 정답 위치 주변에 분포를 생성한다.

    rho_idx, theta_idx: 정답 중심의 행·열 번호
    R, THETA: Heatmap의 전체 행·열 수
    sigma_rho: 행 방향 분포 폭
    sigma_theta: 열 방향 분포 폭
    mode: "gaussian" 또는 "laplace2d"
    kernel_size: 적용할 정사각형 범위. None이면 전체 격자
    """
    if R <= 0 or THETA <= 0:
        raise ValueError("R과 THETA는 양수여야 합니다.")

    if not (0 <= rho_idx < R and 0 <= theta_idx < THETA):
        raise ValueError("정답 중심이 Heatmap 범위를 벗어났습니다.")

    if sigma_rho <= 0 or sigma_theta <= 0:
        raise ValueError(
            "sigma_rho와 sigma_theta는 0보다 커야 합니다."
        )

    if kernel_size is not None:
        if not isinstance(kernel_size, int):
            raise ValueError("kernel_size는 정수 또는 None이어야 합니다.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size는 양의 홀수여야 합니다.")

    rho_grid, theta_grid = torch.meshgrid(
        torch.arange(R, dtype=torch.float32),
        torch.arange(THETA, dtype=torch.float32),
        indexing="ij",
    )

    rho_difference = rho_grid - rho_idx
    theta_difference = theta_grid - theta_idx

    rho_distance = rho_difference / sigma_rho
    theta_distance = theta_difference / sigma_theta

    D = rho_distance**2 + theta_distance**2

    if mode == "gaussian":
        gt_heatmap = torch.exp((-0.5)*D)

    elif mode == "laplace2d":
        gt_heatmap = torch.exp(-torch.sqrt(D))

    else:
        raise ValueError(
            'mode는 "gaussian" 또는 "laplace2d"여야 합니다.'
        )

    if kernel_size is not None:
        radius = kernel_size // 2

        rho_mask = rho_difference.abs() <= radius
        theta_mask = theta_difference.abs() <= radius

        mask = rho_mask & theta_mask

        gt_heatmap = gt_heatmap * mask

    return gt_heatmap


def generate_hough_target(
        annotations, original_w, original_h,
        target_w, target_h, R=112, THETA=100, target_label=1.0,
        sigma_rho=1.0, sigma_theta=1.0, mode="gaussian", kernel_size=5
):
    """
    샘플 하나의 Annotation으로 Gaussian 또는 Laplacian GT Hough Heatmap을 생성한다.

    입력 Annotation 형식:
        [
            {
                "label": 1.0,
                "points": [[x0, y0], [x1, y1]],
            },
            ...
        ]

    target_label:
        지정한 중요도의 직선만 사용한다.
        None이면 모든 중요도의 직선을 사용한다.

    반환:
        float32 Tensor, Shape: (1, R, THETA)
    """
    if min(original_w, original_h, target_w, target_h) <= 1:
        raise ValueError("이미지의 가로·세로 크기는 1보다 커야 합니다.")

    if R <= 0 or THETA <= 0:
        raise ValueError("R과 THETA는 양수여야 합니다.")

    heatmap = torch.zeros(
        (1, R, THETA),
        dtype=torch.float32,
    )

    for annotation in annotations:
        # 지정한 중요도의 Semantic Line만 선택한다.
        if target_label is not None:
            if not math.isclose(
                float(annotation["label"]),
                float(target_label),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                continue

        # 1. 원본 끝점 좌표를 모델 입력 크기로 변환
        resized_points = resize_line_points(
            annotation["points"],
            original_w=original_w,
            original_h=original_h,
            target_w=target_w,
            target_h=target_h,
        )

        # 2. 두 끝점을 Hough 파라미터로 변환
        rho, theta = line_to_hough(
            resized_points,
            image_w=target_w,
            image_h=target_h,
        )

        # 3. Hough 파라미터를 Bin 인덱스로 변환
        rho_idx, theta_idx = hough_to_bin(
            rho,
            theta,
            image_w=target_w,
            image_h=target_h,
            R=R,
            THETA=THETA,
        )

        # 4. Bin 인덱스에 Gaussian or laplacian 적용
        heat_map = make_line_heatmap(
            rho_idx,
            theta_idx,
            R,
            THETA,
            sigma_rho=sigma_rho,
            sigma_theta=sigma_theta,
            mode=mode,
            kernel_size=kernel_size,
        )

        heatmap[0] = torch.maximum(heatmap[0], heat_map)

    return heatmap