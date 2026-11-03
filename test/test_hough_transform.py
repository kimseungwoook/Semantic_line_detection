
from src.hough_dictionary import Multi_scale_HoughDictionary

import torch
import matplotlib.pyplot as plt
import math
import time


# ==============================================
# 공통 설정
# ==============================================

ORIGINAL_W = 224
ORIGINAL_H = 224

R = 112
THETA = 100

FEATURE_SHAPES = [
    (28, 28),
    (14, 14),
    (7, 7)
]


# ==============================================
# STEP 1. 단일 픽셀 Hough Voting 검증
# ==============================================

def test_single_pixel(model):

    print("\n----- STEP 1. 단일 픽셀 Hough Voting 검증 -----")

    feature_map = torch.zeros((1, 1, 14, 14))

    x, y = 7, 7
    feature_map[0, 0, y, x] = 1.0

    hough_feature_maps = model([feature_map])
    hough_result = hough_feature_maps[0][0, 0]

    print("Feature map shape :", feature_map.shape)
    print("Hough feature map shape :", hough_result.shape)
    print("Total votes :", hough_result.sum().item())

    active_theta = (
        hough_result.sum(dim=0) > 0
    ).sum().item()

    print("Active theta bins :", active_theta)

    assert hough_result.sum().item() == THETA
    assert active_theta == THETA

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(
        feature_map[0, 0].numpy(),
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=1
    )

    axes[0].scatter(x, y, color="red", s=50)
    axes[0].set_title("Feature Map (14x14)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    im = axes[1].imshow(
        hough_result.detach().cpu().numpy(),
        cmap="hot",
        origin="lower",
        aspect="auto"
    )

    axes[1].set_title("Hough Space (112x100)")
    axes[1].set_xlabel("Theta Index")
    axes[1].set_ylabel("Rho Index")

    fig.colorbar(im, ax=axes[1], label="Vote")

    plt.tight_layout()
    plt.show()

    print("STEP 1 : PASS")


# ==============================================
# STEP 2. 다중 직선 Hough Voting 검증
# ==============================================

def test_multiple_lines(model):

    print("\n----- STEP 2. 다중 직선 Hough Voting 검증 -----")

    feature_map = torch.zeros((1, 1, 14, 14))

    X = 7
    Y = 7

    # 수평선
    feature_map[0, 0, Y, :] = 1.0

    # 수직선
    feature_map[0, 0, :, X] = 1.0

    # 대각선
    for i in range(14):
        feature_map[0, 0, i, i] = 1.0

    hough_feature_maps = model([feature_map])
    hough_result = hough_feature_maps[0][0, 0]

    active_pixels = torch.count_nonzero(feature_map).item()

    total_votes = hough_result.sum().item()
    max_vote = hough_result.max().item()

    print("Feature map shape :", feature_map.shape)
    print("Hough feature map shape :", hough_result.shape)
    print("Active pixels :", active_pixels)
    print("Total votes :", total_votes)
    print("Max vote :", max_vote)

    assert active_pixels == 40
    assert total_votes == active_pixels * THETA

    # 수직선, 수평선, 대각선의 예상 theta
    expected_theta_indices = {
        "Vertical": 0,
        "Horizontal": 50,
        "Diagonal": 75
    }

    for name, theta_idx in expected_theta_indices.items():

        votes = hough_result[:, theta_idx]
        peak = votes.max().item()
        rho_idx = int(torch.argmax(votes).item())

        print(
            f"{name} | "
            f"Rho index: {rho_idx} | "
            f"Theta index: {theta_idx} | "
            f"Peak vote: {peak}"
        )

        assert peak >= 14

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(
        feature_map[0, 0].numpy(),
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Multiple Lines (14x14)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    im = axes[1].imshow(
        hough_result.detach().cpu().numpy(),
        cmap="hot",
        origin="lower",
        aspect="auto"
    )

    axes[1].set_title("Hough Space (112x100)")
    axes[1].set_xlabel("Theta Index")
    axes[1].set_ylabel("Rho Index")

    fig.colorbar(im, ax=axes[1], label="Vote")

    plt.tight_layout()
    plt.show()

    print("STEP 2 : PASS")


# ==============================================
# STEP 3. Multi-Scale Hough Transform 검증
# ==============================================

def test_multiscale_hough(model):

    print("\n----- STEP 3. Multi-Scale Hough Transform 검증 -----")

    feature_maps = []

    for H, W in FEATURE_SHAPES:

        fmap = torch.zeros((1, 1, H, W))

        # 동일한 원본 좌표에 대응하는 수직선
        fmap[0, 0, :, 0] = 1.0

        feature_maps.append(fmap)

    hough_feature_maps = model(feature_maps)

    hough_results = [
        result[0, 0]
        for result in hough_feature_maps
    ]

    peak_positions = []

    for idx, result in enumerate(hough_results):

        max_vote = result.max().item()

        peak_idx = torch.argmax(result)
        rho_idx = int((peak_idx // THETA).item())
        theta_idx = int((peak_idx % THETA).item())

        peak_positions.append((rho_idx, theta_idx))

        print(f"\nScale {idx + 1}")
        print("Feature map shape :", feature_maps[idx].shape)
        print("Max vote :", max_vote)
        print("Peak Rho index :", rho_idx)
        print("Peak Theta index :", theta_idx)
        print("Peak Theta :", theta_idx * 180 / THETA)

        expected_votes = FEATURE_SHAPES[idx][0]

        assert max_vote == expected_votes
        assert rho_idx == 16
        assert theta_idx == 0

    assert len(set(peak_positions)) == 1

    # 시각화: Figure는 반복문 밖에서 한 번만 생성
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(3):

        # Feature Map
        axes[0, i].imshow(
            feature_maps[i][0, 0].numpy(),
            cmap="gray",
            origin="upper",
            vmin=0,
            vmax=1
        )

        H, W = feature_maps[i].shape[2:]

        axes[0, i].set_title(f"Feature Map ({H}x{W})")
        axes[0, i].set_xlabel("X")
        axes[0, i].set_ylabel("Y")

        # Hough Space
        im = axes[1, i].imshow(
            hough_results[i].detach().cpu().numpy(),
            cmap="hot",
            origin="lower",
            aspect="auto"
        )

        axes[1, i].set_title("Hough Space (112x100)")
        axes[1, i].set_xlabel("Theta Index")
        axes[1, i].set_ylabel("Rho Index")

        fig.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.show()

    print("\nSTEP 3 : PASS")


# ==============================================
# STEP 4. 좌표 변환 및 양자화 오차 검증
# ==============================================

def test_coordinate_quantization(model):

    print("\n----- STEP 4. 좌표 변환 및 양자화 오차 검증 -----")

    center_x = (ORIGINAL_W - 1) / 2
    center_y = (ORIGINAL_H - 1) / 2

    rho_max = math.sqrt(center_x**2 + center_y**2)

    rho_step = (2 * rho_max) / R
    theta_step = 180 / THETA

    print(f"Rho max : {rho_max:.6f}")
    print(f"Rho step : {rho_step:.6f} px")
    print(f"Theta step : {theta_step:.6f} degrees")
    print(f"Max theoretical rho error : {rho_step/2:.6f} px")
    print(f"Max theoretical theta error : {theta_step/2:.6f} degrees")

    # 실제 Dictionary와 수학적 계산 비교
    for scale_idx, (H, W) in enumerate(FEATURE_SHAPES):

        dictionary = getattr(
            model, f"dictionary_{scale_idx}"
        )

        max_rho_error = 0.0

        test_points = [
            (0, 0),
            (W - 1, 0),
            (0, H - 1),
            (W - 1, H - 1),
            (W // 2, H // 2)
        ]

        for x, y in test_points:

            original_x = x * (ORIGINAL_W - 1) / (W - 1)
            original_y = y * (ORIGINAL_H - 1) / (H - 1)

            x_centered = original_x - center_x
            y_centered = original_y - center_y

            point_idx = y * W + x

            for theta_idx in range(THETA):

                angle = math.radians(
                    theta_idx * theta_step
                )

                rho = (
                    x_centered * math.cos(angle)
                    + y_centered * math.sin(angle)
                )

                # 기존 Hough Dictionary의 양자화 방식
                expected_idx = math.floor(
                    (rho + rho_max) / rho_step
                )
                expected_idx = max(
                    0, min(R - 1, expected_idx)
                )

                actual_idx = int(
                    torch.argmax(
                        dictionary[point_idx, :, theta_idx]
                    ).item()
                )

                # float32와 경계값 반올림으로 인해
                # 경계 부근에서 1 bin 차이가 날 수 있음
                assert abs(expected_idx - actual_idx) <= 1, (
                    f"Scale {scale_idx+1}, "
                    f"Point ({x},{y}), Theta {theta_idx}: "
                    f"Expected {expected_idx}, Got {actual_idx}"
                )

                reconstructed_rho = (
                    -rho_max + (actual_idx + 0.5) * rho_step
                )

                error = abs(rho - reconstructed_rho)
                max_rho_error = max(max_rho_error, error)

        print(
            f"Scale {scale_idx+1} ({H}x{W}) | "
            f"Max rho error : {max_rho_error:.6f} px"
        )

        # 수치적 경계 오차를 제외하면 반 bin 이내
        assert max_rho_error <= rho_step * 1.5 + 1e-4

    # Theta 양자화
    print("\nTheta quantization:")

    for true_theta in [0, 23, 45, 90, 135, 179]:

        theta_idx = round(true_theta / theta_step) % THETA
        recovered_theta = theta_idx * theta_step

        error = abs(true_theta - recovered_theta)
        error = min(error, 180 - error)

        print(
            f"True={true_theta:.2f} | "
            f"Recovered={recovered_theta:.2f} | "
            f"Error={error:.4f}"
        )

        assert error <= theta_step / 2 + 1e-6

    print("\nSTEP 4 : PASS")


# ==============================================
# STEP 5. GPU 및 역전파 검증
# ==============================================

def test_gradient(model):

    print("\n----- STEP 5-1. Gradient 검증 -----")

    model = model.cpu()

    feature_maps = [
        torch.randn(
            2, 3, H, W,
            requires_grad=True
        )
        for H, W in FEATURE_SHAPES
    ]

    outputs = model(feature_maps)

    loss = sum(
        output.square().mean()
        for output in outputs
    )

    loss.backward()

    for idx, fmap in enumerate(feature_maps):

        assert fmap.grad is not None
        assert torch.isfinite(fmap.grad).all()
        assert fmap.grad.abs().sum().item() > 0

        print(
            f"Scale {idx+1} | "
            f"Gradient Mean : "
            f"{fmap.grad.abs().mean().item():.8f}"
        )

    # Dictionary는 학습되지 않는 Buffer
    for idx in range(len(FEATURE_SHAPES)):

        dictionary = getattr(
            model, f"dictionary_{idx}"
        )

        assert not dictionary.requires_grad

    print("STEP 5-1 : PASS")


def test_gpu(model):

    print("\n----- STEP 5-2. GPU 검증 -----")

    if not torch.cuda.is_available():
        print("CUDA unavailable : SKIP")
        return

    feature_maps = [
        torch.randn(1, 2, H, W)
        for H, W in FEATURE_SHAPES
    ]

    model = model.cpu()

    with torch.no_grad():
        cpu_outputs = model(feature_maps)

    model = model.cuda()

    gpu_inputs = [
        fmap.cuda()
        for fmap in feature_maps
    ]

    with torch.no_grad():
        gpu_outputs = model(gpu_inputs)

    for idx, (cpu_out, gpu_out) in enumerate(
        zip(cpu_outputs, gpu_outputs)
    ):

        torch.testing.assert_close(
            cpu_out,
            gpu_out.cpu(),
            rtol=1e-4,
            atol=1e-4
        )

        print(f"Scale {idx+1} | CPU/GPU PASS")

    # GPU Gradient 검증
    gpu_inputs = [
        torch.randn(
            1, 2, H, W,
            device="cuda",
            requires_grad=True
        )
        for H, W in FEATURE_SHAPES
    ]

    outputs = model(gpu_inputs)

    loss = sum(
        output.square().mean()
        for output in outputs
    )

    loss.backward()

    for idx, fmap in enumerate(gpu_inputs):

        assert fmap.grad is not None
        assert torch.isfinite(fmap.grad).all()
        assert fmap.grad.abs().sum().item() > 0

        print(f"Scale {idx+1} | GPU Gradient PASS")

    model.cpu()
    torch.cuda.empty_cache()

    print("STEP 5-2 : PASS")


# ==============================================
# STEP 6. 메모리 및 연산 성능 검증
# ==============================================

def test_memory(model):

    print("\n----- STEP 6-1. Dictionary 메모리 검증 -----")

    total_dense_memory = 0
    total_sparse_estimate = 0

    for idx, (H, W) in enumerate(FEATURE_SHAPES):

        dictionary = getattr(
            model, f"dictionary_{idx}"
        )

        dense_bytes = (
            dictionary.numel()
            * dictionary.element_size()
        )

        nonzero = torch.count_nonzero(dictionary).item()

        # CSR, COO 중 최적 저장 방식을 결정하기 전
        # 2D Sparse COO 기준 추정
        sparse_bytes = nonzero * (8 * 2 + 4)

        total_dense_memory += dense_bytes
        total_sparse_estimate += sparse_bytes

        print(f"\nScale {idx+1} ({H}x{W})")
        print("Dictionary Shape :", dictionary.shape)
        print("Nonzero elements :", nonzero)
        print(f"Dense memory : {dense_bytes / 1024**2:.3f} MiB")
        print(f"Sparse estimate : {sparse_bytes / 1024**2:.3f} MiB")

        assert nonzero == H * W * THETA

    print(
        f"\nTotal Dense Memory : "
        f"{total_dense_memory / 1024**2:.3f} MiB"
    )

    print(
        f"Total Sparse Estimate : "
        f"{total_sparse_estimate / 1024**2:.3f} MiB"
    )

    print("STEP 6-1 : PASS")


def test_forward_speed(model, device="cpu"):

    print(f"\n----- STEP 6-2. Forward Speed ({device}) -----")

    model = model.to(device)
    model.eval()

    feature_maps = [
        torch.randn(2, 8, H, W, device=device)
        for H, W in FEATURE_SHAPES
    ]

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():

        # Warmup
        for _ in range(3):
            model(feature_maps)

        synchronize()

        repetitions = 10

        start = time.perf_counter()

        for _ in range(repetitions):
            outputs = model(feature_maps)

        synchronize()

        elapsed = time.perf_counter() - start

    avg_time = (elapsed / repetitions) * 1000

    print(f"Average Forward Time : {avg_time:.3f} ms")

    for idx, output in enumerate(outputs):
        print(f"Scale {idx+1} output : {output.shape}")

    print(f"STEP 6-2 ({device}) : PASS")

    return avg_time


# ==============================================
# 메인 실행
# ==============================================

if __name__ == "__main__":

    print("=" * 50)
    print("PHASE 2. Multi-Scale Hough Dictionary Validation")
    print("=" * 50)

    # STEP 1, 2용 Dictionary
    single_model = Multi_scale_HoughDictionary(
        original_w=ORIGINAL_W,
        original_h=ORIGINAL_H,
        feature_shapes=[(14, 14)],
        R=R,
        THETA=THETA
    )

    # STEP 3~6용 Multi-Scale Dictionary
    multiscale_model = Multi_scale_HoughDictionary(
        original_w=ORIGINAL_W,
        original_h=ORIGINAL_H,
        feature_shapes=FEATURE_SHAPES,
        R=R,
        THETA=THETA
    )

    # STEP 1
    test_single_pixel(single_model)

    # STEP 2
    test_multiple_lines(single_model)

    # STEP 3
    test_multiscale_hough(multiscale_model)

    # STEP 4
    test_coordinate_quantization(multiscale_model)

    # STEP 5
    test_gradient(multiscale_model)
    test_gpu(multiscale_model)

    # STEP 6
    test_memory(multiscale_model)

    cpu_time = test_forward_speed(
        multiscale_model,
        device="cpu"
    )

    if torch.cuda.is_available():

        gpu_time = test_forward_speed(
            multiscale_model,
            device="cuda"
        )

        print(
            f"\nCPU/GPU Forward Speed Ratio : "
            f"{cpu_time / gpu_time:.2f}x"
        )

    print("\n" + "=" * 50)
    print("PHASE 2 VALIDATION COMPLETED")
    print("=" * 50)
