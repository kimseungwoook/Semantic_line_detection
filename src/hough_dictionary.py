import numpy as np
import torch
import matplotlib.pyplot as plt

def hough_transform(H, W, x, y, R, THETA):
    """
    :param H: event_count_map Height
    :param W: event_count_map Width
    :param x: 특정 x 좌표
    :param y: 특정 y 좌표
    :param R: 허프 공간 R
    :param THETA: 허프 공간 THETA

    :return: Hough_space : 해당 좌표가 허프 공간에서 이루는 곡선 형태 행렬 반환 [R, THETA]
    """

    # 입력 event_count_map 생성
    event_count_map = torch.zeros((H, W))

    # 특정 픽셀 선택
    event_count_map[y, x] = 1

    # 이미지 중심을 원점으로 변환
    center_x = (W - 1)  / 2
    center_y = (H - 1) / 2

    # 이미지 중심으로 하였을 때 특정 좌표
    x_ = x - center_x
    y_ = y - center_y

    # 허프 공간 생성
    hough_space = torch.zeros((R, THETA))

    # 각도 생성
    angle = torch.arange(0, THETA, dtype=torch.float32)
    rad = torch.deg2rad(angle)

    # 거리 범위 및 거리 간격 계산 약 3.38px
    value = center_x**2 + center_y**2
    rho_max = torch.sqrt(torch.tensor(value, dtype=torch.float32))
    rho_step = (rho_max * 2) / R

    # 허프 곡선 계산
    rho = x_ * torch.cos(rad) + y_ * torch.sin(rad)

    # 실제 거리를 허프 공간의 행 번호로 변환
    rho_idx = torch.floor((rho_max + rho) / rho_step).long()
    rho_idx = torch.clamp(rho_idx, 0, R - 1)

    # hough space에 적용
    hough_space[rho_idx, angle.long()] += event_count_map[y, x]

    return hough_space

def create_hough_dictionary(H, W, R, THETA):
    hough_dictionary = torch.zeros((H*W, R, THETA))
    print(f"hough_dictionary shape : {hough_dictionary.shape}")

    HW_count = 0
    for y in range(H):
        for x in range(W):
            hough_space = hough_transform(H, W, x, y, R, THETA)
            hough_dictionary[HW_count] = hough_space
            HW_count += 1

    print(f"Hough_dictionary shape : {hough_dictionary.shape}")
    print(hough_dictionary[0].sum())
    print(hough_dictionary[8].sum())
    return hough_dictionary

if __name__ == "__main__":
    H = 5
    W = 5
    x = 223
    y = 130
    R = 128
    THETA = 180
    hough_dictionary = create_hough_dictionary(H, W, R, THETA)

    fig, axes = plt.subplots(H, W, figsize=(12, 12))

    for y in range(H):
        for x in range(W):
            idx = y * W + x

            axes[y, x].imshow(
                hough_dictionary[idx].cpu().numpy(),
                cmap="hot",
                origin="lower",
                aspect="auto"
            )
            axes[y, x].set_title(f"pixel idx={idx} (x={x}, y={y})")
            axes[y, x].set_xlabel("Theta")
            axes[y, x].set_ylabel("Rho")

    plt.tight_layout()
    plt.show()

    # hough_space = hough_transform(H, W, x, y, R, THETA)
    #
    # event_count_map = np.zeros((H, W))
    # event_count_map[y, x] = 1
    #
    # # 시각화
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #
    # # 1. 입력 이미지
    # axes[0].imshow(event_count_map, cmap="gray")
    # axes[0].scatter(x, y, color="red", s=50)
    # axes[0].set_title("Event Count Map")
    # axes[0].set_xlabel("X")
    # axes[0].set_ylabel("Y")
    #
    # # 2. 허프 공간
    # axes[1].imshow(
    #     hough_space,
    #     cmap="hot",
    #     origin="lower",
    #     aspect="auto",
    #     extent=[0, 180, -216.4, 216.4]
    # )
    # axes[1].set_title("Hough Space")
    # axes[1].set_xlabel("Theta (degree)")
    # axes[1].set_ylabel("Rho (pixel)")
    #
    # plt.tight_layout()
    # plt.show()