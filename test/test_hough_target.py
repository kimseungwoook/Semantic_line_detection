import math
from itertools import combinations

import torch
import matplotlib.pyplot as plt

from src.dataset import EventDataset
from src.hough_target import (
    generate_hough_target,
    make_line_heatmap,
    resize_line_points,
    line_to_hough,
    hough_to_bin,
)


def validate_distribution():
    """Gaussian·Laplacian의 값, 범위, 겹침 처리를 확인한다."""
    for mode in ("gaussian", "laplace2d"):
        heatmap = make_line_heatmap(
            rho_idx=53,
            theta_idx=54,
            R=112,
            THETA=100,
            sigma_rho=1.0,
            sigma_theta=1.0,
            mode=mode,
            kernel_size=5,
        )

        assert heatmap.shape == (112, 100)
        assert heatmap.dtype == torch.float32
        assert torch.isfinite(heatmap).all().item()
        assert ((heatmap >= 0) & (heatmap <= 1)).all().item()

        # 정답 중심은 1
        assert heatmap[53, 54].item() == 1.0

        # 중심에서 열 방향으로 한 칸 떨어진 값
        expected = (
            math.exp(-0.5)
            if mode == "gaussian"
            else math.exp(-1.0)
        )

        assert math.isclose(
            heatmap[53, 55].item(),
            expected,
            rel_tol=1e-5,
        )

        # 중심에서 3칸 떨어진 곳은 5×5 범위 밖
        assert heatmap[53, 57].item() == 0.0
        assert heatmap[56, 54].item() == 0.0

        # 중심이 경계에 없으므로 25칸이 활성화
        assert torch.count_nonzero(heatmap).item() == 25

        print(
            f"{mode}: 중심={heatmap[53, 54].item():.3f}, "
            f"한 칸 옆={heatmap[53, 55].item():.6f}, "
            f"범위 밖={heatmap[53, 57].item():.1f} | PASS"
        )

        # 실제 Annotation 두 개로 최댓값 결합 검증
        annotation_a = {
            "label": 1.0,
            "points": [[100, 0], [100, 223]],
        }
        annotation_b = {
            "label": 1.0,
            "points": [[106, 0], [106, 223]],
        }

        settings = dict(
            original_w=224,
            original_h=224,
            target_w=224,
            target_h=224,
            R=112,
            THETA=100,
            target_label=1.0,
            sigma_rho=1.0,
            sigma_theta=1.0,
            mode=mode,
            kernel_size=5,
        )

        target_a = generate_hough_target(
            [annotation_a], **settings
        )
        target_b = generate_hough_target(
            [annotation_b], **settings
        )
        combined = generate_hough_target(
            [annotation_a, annotation_b], **settings
        )

        assert combined.shape == (1, 112, 100)
        assert torch.equal(
            combined,
            torch.maximum(target_a, target_b),
        )

        # 두 분포가 실제로 겹치는 사례인지도 확인
        assert ((target_a > 0) & (target_b > 0)).any().item()

        # 중요도가 다른 Annotation은 제외
        ignored = {
            "label": 0.5,
            "points": [[20, 0], [20, 223]],
        }
        filtered = generate_hough_target(
            [annotation_a, ignored], **settings
        )
        assert torch.equal(filtered, target_a)

        # 정답이 없으면 전체 0
        empty = generate_hough_target([], **settings)
        assert empty.shape == (1, 112, 100)
        assert torch.count_nonzero(empty).item() == 0

        print(f"{mode}: 최댓값 결합·중요도 필터·빈 GT | PASS")


def get_boundary_points(a, b, c, width, height):
    """직선 ax + by = c와 이미지 경계의 교점을 구한다."""
    points = []
    eps = 1e-8

    def add_point(x, y):
        if (
            -eps <= x <= width - 1 + eps
            and -eps <= y <= height - 1 + eps
        ):
            x = min(max(x, 0.0), width - 1)
            y = min(max(y, 0.0), height - 1)

            for px, py in points:
                if math.hypot(x - px, y - py) < eps:
                    return

            points.append((x, y))

    if abs(b) > eps:
        add_point(0, c / b)
        add_point(width - 1, (c - a * (width - 1)) / b)

    if abs(a) > eps:
        add_point(c / a, 0)
        add_point((c - b * (height - 1)) / a, height - 1)

    if len(points) < 2:
        return None

    return max(
        combinations(points, 2),
        key=lambda pair: (
            (pair[0][0] - pair[1][0]) ** 2
            + (pair[0][1] - pair[1][1]) ** 2
        ),
    )


def restore_line_from_bin(
    rho_idx,
    theta_idx,
    original_w,
    original_h,
    target_w,
    target_h,
    R,
    THETA,
):
    """Hough Bin을 원본 이미지 좌표계의 직선으로 복원한다."""
    center_x = (target_w - 1) / 2
    center_y = (target_h - 1) / 2

    rho_max = math.hypot(center_x, center_y)
    rho_step = 2 * rho_max / R

    # floor로 양자화한 구간의 중앙값으로 복원
    rho = -rho_max + (rho_idx + 0.5) * rho_step
    theta = theta_idx * math.pi / THETA

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # 현재 Resize 규칙을 역으로 적용한 원본 직선
    a = cos_theta * target_w / original_w
    b = sin_theta * target_h / original_h
    c = rho + center_x * cos_theta + center_y * sin_theta

    return get_boundary_points(
        a, b, c, original_w, original_h
    )


if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    threshold = 3
    sample_idx = 0

    target_w = 224
    target_h = 224
    R = 112
    THETA = 100
    target_label = 1.0

    # 시각화할 분포 선택
    mode = "gaussian"  # 또는 "laplace2d"
    sigma_rho = 1.0
    sigma_theta = 1.0
    kernel_size = 5

    validate_distribution()

    dataset = EventDataset(root_dir, threshold)
    sample = dataset[sample_idx]

    count_map = sample["count_map"][0].float()
    original_h, original_w = count_map.shape

    original_image = (
        count_map.clamp(0, threshold) / threshold
    ).cpu().numpy()

    gt_heatmap = generate_hough_target(
        annotations=sample["annotation"],
        original_w=original_w,
        original_h=original_h,
        target_w=target_w,
        target_h=target_h,
        R=R,
        THETA=THETA,
        target_label=target_label,
        sigma_rho=sigma_rho,
        sigma_theta=sigma_theta,
        mode=mode,
        kernel_size=kernel_size,
    )

    assert gt_heatmap.shape == (1, R, THETA)
    assert torch.isfinite(gt_heatmap).all().item()
    assert ((gt_heatmap >= 0) & (gt_heatmap <= 1)).all().item()

    target_annotations = [
        annotation
        for annotation in sample["annotation"]
        if math.isclose(
            float(annotation["label"]),
            target_label,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ]

    # 현재 GT는 중심값이 1이므로 그 위치만 추출한다.
    peak_bins = torch.nonzero(
        gt_heatmap[0] == 1.0,
        as_tuple=False,
    ).tolist()

    # Annotation에서 계산한 중심 Bin과 일치하는지 확인한다.
    expected_bins = set()

    for annotation in target_annotations:
        points = resize_line_points(
            annotation["points"],
            original_w,
            original_h,
            target_w,
            target_h,
        )
        rho, theta = line_to_hough(
            points, target_w, target_h
        )
        expected_bins.add(
            hough_to_bin(
                rho, theta, target_w, target_h, R, THETA
            )
        )

    assert {tuple(p) for p in peak_bins} == expected_bins

    fig, (ax_original, ax_hough) = plt.subplots(
        1, 2, figsize=(15, 6)
    )

    ax_original.imshow(
        original_image,
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="upper",
    )

    for idx, annotation in enumerate(target_annotations):
        (x0, y0), (x1, y1) = annotation["points"]

        ax_original.plot(
            [x0, x1],
            [y0, y1],
            color="lime",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Original annotation" if idx == 0 else None,
        )

    restored_count = 0

    for rho_idx, theta_idx in peak_bins:
        restored = restore_line_from_bin(
            rho_idx,
            theta_idx,
            original_w,
            original_h,
            target_w,
            target_h,
            R,
            THETA,
        )

        if restored is None:
            print(f"복원 불가 Bin: ({rho_idx}, {theta_idx})")
            continue

        (x0, y0), (x1, y1) = restored

        ax_original.plot(
            [x0, x1],
            [y0, y1],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=(
                "Restored from GT peak"
                if restored_count == 0 else None
            ),
        )

        restored_count += 1
        print(
            f"Peak ({rho_idx}, {theta_idx})"
            f" → ({x0:.2f}, {y0:.2f}), ({x1:.2f}, {y1:.2f})"
        )

    ax_original.set_xlim(-0.5, original_w - 0.5)
    ax_original.set_ylim(original_h - 0.5, -0.5)
    ax_original.set_aspect("equal")
    ax_original.set_title("Annotation vs. Restored GT Peaks")
    ax_original.set_xlabel("Original x")
    ax_original.set_ylabel("Original y")

    if target_annotations or restored_count:
        ax_original.legend()

    image = ax_hough.imshow(
        gt_heatmap[0].cpu().numpy(),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    if peak_bins:
        ax_hough.scatter(
            [p[1] for p in peak_bins],
            [p[0] for p in peak_bins],
            color="red",
            marker="+",
            label="GT centers",
        )
        ax_hough.legend()

    ax_hough.set_xlabel("Theta bin")
    ax_hough.set_ylabel("Rho bin")
    ax_hough.set_title(
        f"{mode} | sigma=({sigma_rho}, {sigma_theta})"
        f" | kernel={kernel_size}"
    )
    fig.colorbar(image, ax=ax_hough, label="Target")

    fig.suptitle(
        f'{sample["folder_name"]} / {sample["file_name"]}'
    )
    plt.tight_layout()

    print(f"정답 Annotation 개수: {len(target_annotations)}")
    print(f"중심 Peak 개수: {len(peak_bins)}")
    print(f"0보다 큰 Bin 개수: {torch.count_nonzero(gt_heatmap).item()}")
    print(f"복원 직선 개수: {restored_count}")
    print("GT 분포·중심 검증 PASS")

    plt.show()