import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from src.hough_dictionary import create_hough_dictionary


if __name__ == "__main__":
    B = 1
    C = 1
    H = 50
    W = 50
    R = 128
    THETA = 180

    # raw voting에 적용할 임계값
    vote_threshold = 20.0

    # 1. 테스트 이미지 생성
    event_image = torch.zeros(B, C, H, W, dtype=torch.float32)

    # 가로선
    event_image[0, 0, H // 2, :] = 1.0

    # 세로선
    event_image[0, 0, :, W // 2] = 1.0

    # 좌상단 → 우하단 대각선: 직사각형 이미지에도 적용
    n = max(H, W)
    x = torch.linspace(0, W - 1, n).round().long()
    y = torch.linspace(0, H - 1, n).round().long()
    event_image[0, 0, y, x] = 1.0

    # 2. 허프 변환
    hough_dictionary = create_hough_dictionary(H, W, R, THETA).float()

    event_flat = event_image.flatten(2)              # [B, C, H*W]
    dictionary = hough_dictionary.reshape(H * W, R * THETA)

    hough_feature = event_flat @ dictionary
    hough_feature = hough_feature.reshape(B, C, R, THETA)

    print(f"hough_feature shape: {hough_feature.shape}")

    input_map = event_image[0, 0].detach().cpu().numpy()
    hough_map = hough_feature[0, 0].detach().cpu().numpy()

    # 3. 임계값을 넘는 bin들을 연결 영역으로 묶기
    # 학습된 예측값이 아닌 raw voting이므로 sigmoid는 적용하지 않음
    binary_map = hough_map > vote_threshold
    component_map = label(binary_map, connectivity=1)
    regions = regionprops(component_map)

    print(f"Detected regions: {len(regions)}")

    # 4. 사전과 동일한 거리 좌표계
    center_x = (W - 1) / 2
    center_y = (H - 1) / 2
    rho_max = np.hypot(center_x, center_y)
    rho_step = 2 * rho_max / R

    # 5. 입력 / 허프 공간 / 검출 직선 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(input_map, cmap="gray")
    axes[0].set_title("Input: Three Lines")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    im = axes[1].imshow(
        hough_map,
        cmap="hot",
        origin="lower",
        aspect="auto",
    )
    axes[1].set_title(f"Hough Voting > {vote_threshold}")
    axes[1].set_xlabel("Theta bin")
    axes[1].set_ylabel("Rho bin")
    fig.colorbar(im, ax=axes[1], label="Accumulated vote")

    axes[2].imshow(input_map, cmap="gray")
    axes[2].set_title("Detected Lines")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    for i, region in enumerate(regions):
        # 연결 영역 중심: 최대값 위치가 아닌 평균 좌표
        rho_idx, theta_idx = region.centroid

        # 현재 create_hough_dictionary는 각도 간격 1도 사용
        theta = np.deg2rad(theta_idx)

        # floor 방식으로 만든 거리 bin의 중심을 대표 거리로 사용
        rho = -rho_max + (rho_idx + 0.5) * rho_step

        color = plt.get_cmap("tab10")(i % 10)

        print(
            f"Line {i + 1}: "
            f"R bin={rho_idx:.2f}, "
            f"Theta={theta_idx:.2f}°, "
            f"rho={rho:.3f}"
        )

        # 허프 공간에 대표 좌표 표시
        axes[1].scatter(
            theta_idx, rho_idx,
            color=color, marker="x", s=60,
        )

        # 이미지 중심 기준: 직선 위의 기준점
        x0 = center_x + rho * np.cos(theta)
        y0 = center_y + rho * np.sin(theta)

        # 직선의 접선 방향
        dx = -np.sin(theta)
        dy = np.cos(theta)
        length = np.hypot(H, W)

        # bin 중심에 대응하는 대표 직선 표시
        axes[2].plot(
            [x0 - length * dx, x0 + length * dx],
            [y0 - length * dy, y0 + length * dy],
            color=color,
            linewidth=1.5,
            label=f"{i + 1}: theta={theta_idx:.1f}°",
        )

    axes[2].set_xlim(-0.5, W - 0.5)
    axes[2].set_ylim(H - 0.5, -0.5)

    if regions:
        axes[2].legend(fontsize=8)

    # 0°와 179° 부근의 영역은 별개로 처리되므로
    # 같은 직선이 중복 검출될 수 있음.
    plt.tight_layout()
    plt.show()