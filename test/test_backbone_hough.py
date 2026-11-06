from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.dataset import EventDataset
from src.event_backbone import EventBackbone
from src.hough_dictionary import Multi_scale_HoughDictionary


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward(fn, device, warmup=3, repeats=10):
    """워밍업 이후 Forward 시간 측정. 출력 검증 시간은 포함하지 않는다."""
    for _ in range(warmup):
        output = fn()

    del output
    synchronize(device)

    elapsed_ms = []

    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()

        output = fn()

        synchronize(device)
        elapsed_ms.append((perf_counter() - start) * 1000)
        del output

    return {
        "mean": sum(elapsed_ms) / len(elapsed_ms),
        "min": min(elapsed_ms),
        "max": max(elapsed_ms),
    }


def print_timing(name, result):
    print(
        f"{name}: "
        f"mean={result['mean']:.3f} ms, "
        f"min={result['min']:.3f} ms, "
        f"max={result['max']:.3f} ms"
    )


def mib(num_bytes):
    return num_bytes / (1024 ** 2)


if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    threshold = 3
    batch_size = 1

    R = 112
    THETA = 100

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    dataset = EventDataset(root_dir, threshold)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    sample = next(iter(dataloader))
    event_input = sample["event_input"].to(device)

    model = EventBackbone().to(device)
    model.eval()

    B, C, original_h, original_w = event_input.shape
    assert (C, original_h, original_w) == (1, 224, 224)
    assert torch.isfinite(event_input).all().item()

    expected_feature_shapes = [
        (B, 256, 56, 56),
        (B, 512, 28, 28),
        (B, 1024, 14, 14),
    ]

    expected_hough_shapes = [
        (B, 256, R, THETA),
        (B, 512, R, THETA),
        (B, 1024, R, THETA),
    ]

    with torch.no_grad():
        # 1. Backbone 출력 검증
        feature_maps = model(event_input)

        assert len(feature_maps) == len(expected_feature_shapes)

        for idx, fmap in enumerate(feature_maps):
            assert tuple(fmap.shape) == expected_feature_shapes[idx]
            assert torch.isfinite(fmap).all().item()
            print(f"Feature Map {idx}: {tuple(fmap.shape)}, PASS")

        feature_shapes = [
            tuple(fmap.shape[2:]) for fmap in feature_maps
        ]

        # 2. CPU Dictionary 생성 시간
        start = perf_counter()

        hough_dictionary = Multi_scale_HoughDictionary(
            original_w,
            original_h,
            feature_shapes,
            R,
            THETA,
            operation=model.get_coordinate_operation()
        )

        construction_ms = (perf_counter() - start) * 1000
        print(f"\nDictionary CPU 생성: {construction_ms:.3f} ms")

        dictionary_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in hough_dictionary.buffers()
        )
        print(f"Dictionary Buffer 메모리: {mib(dictionary_bytes):.3f} MiB")

        for idx in range(3):
            coord = hough_dictionary.previous_center(1, 1, hough_dictionary.scale_operation[idx])
            print(f"Scale : {idx} : {coord}")


        # 3. Device 이동 시간 — Dictionary 생성 시간과 분리
        synchronize(device)
        start = perf_counter()

        hough_dictionary = hough_dictionary.to(device)
        hough_dictionary.eval()

        synchronize(device)
        transfer_ms = (perf_counter() - start) * 1000
        print(f"Dictionary Device 이동: {transfer_ms:.3f} ms")

        # 4. Hough 출력 Shape 및 NaN/Inf 검증
        hough_feature_maps = hough_dictionary(feature_maps)

        assert len(hough_feature_maps) == len(expected_hough_shapes)

        print("\nHough Feature 분포")
        print("전체 Batch·Channel·Rho·Theta 원소에 대한 통계입니다.")

        output_bytes = 0

        for idx, hmap in enumerate(hough_feature_maps):
            assert tuple(hmap.shape) == expected_hough_shapes[idx]
            assert torch.isfinite(hmap).all().item()

            tensor_bytes = hmap.numel() * hmap.element_size()
            output_bytes += tensor_bytes

            print(f"\nScale {feature_shapes[idx]}: {tuple(hmap.shape)}, PASS")
            print(f"  min  : {hmap.min().item():.6f}")
            print(f"  max  : {hmap.max().item():.6f}")
            print(f"  mean : {hmap.mean().item():.6f}")
            print(f"  std  : {hmap.std(unbiased=False).item():.6f}")
            print(f"  메모리: {mib(tensor_bytes):.3f} MiB")

        print(f"\nHough 출력 합계: {mib(output_bytes):.3f} MiB")

        # 측정 중 기존 출력 Tensor가 추가로 남지 않도록 해제
        del hmap
        del hough_feature_maps

        # 5. Forward 시간 측정
        backbone_timing = measure_forward(
            lambda: model(event_input),
            device,
        )

        hough_timing = measure_forward(
            lambda: hough_dictionary(feature_maps),
            device,
        )

        # 고정 Feature Map을 해제한 뒤 전체 연결 측정
        del fmap
        del feature_maps

        full_forward = lambda: hough_dictionary(model(event_input))

        full_timing = measure_forward(
            full_forward,
            device,
        )

        print("\nForward 시간 — 워밍업 3회, 측정 10회")
        print_timing("Backbone", backbone_timing)
        print_timing("Hough Transform", hough_timing)
        print_timing("Backbone + Hough", full_timing)

        # 6. 워밍업 이후 전체 Forward의 GPU 메모리 측정
        if device.type == "cuda":
            synchronize(device)

            baseline_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

            outputs = full_forward()
            synchronize(device)

            peak_bytes = torch.cuda.max_memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)

            print("\nGPU 메모리 — 전체 Forward 1회")
            print(f"  실행 전 할당: {mib(baseline_bytes):.3f} MiB")
            print(f"  최대 할당   : {mib(peak_bytes):.3f} MiB")
            print(
                f"  최대 추가 할당: "
                f"{mib(peak_bytes - baseline_bytes):.3f} MiB"
            )
            print(f"  캐시 포함 예약: {mib(reserved_bytes):.3f} MiB")

            del outputs

    print("\nSTEP 3-5 Forward 검증 PASS")