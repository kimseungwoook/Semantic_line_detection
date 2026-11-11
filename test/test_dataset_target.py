import math

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import EventDataset
from src.hough_target import generate_hough_target


TARGET_W = 224
TARGET_H = 224
R = 112
THETA = 100
TARGET_LABEL = 1.0

# 현재 Dataset의 GT 생성 설정과 같아야 한다.
MODE = "gaussian"
SIGMA_RHO = 1.0
SIGMA_THETA = 1.0
KERNEL_SIZE = 5


def collate_fn(batch):
    """Tensor는 쌓고 Annotation은 리스트로 유지한다."""
    return {
        "event_input": torch.stack([
            sample["event_input"] for sample in batch
        ]),
        "gt_heatmap": torch.stack([
            sample["gt_heatmap"] for sample in batch
        ]),
        "annotation": [
            sample["annotation"] for sample in batch
        ],
        "file_name": [
            sample["file_name"] for sample in batch
        ],
        "folder_name": [
            sample["folder_name"] for sample in batch
        ],
    }


def validate_sample(sample, idx):
    event_input = sample["event_input"]
    gt_heatmap = sample["gt_heatmap"]

    assert tuple(event_input.shape) == (1, TARGET_H, TARGET_W)
    assert tuple(gt_heatmap.shape) == (1, R, THETA)

    assert event_input.dtype == torch.float32
    assert gt_heatmap.dtype == torch.float32

    assert torch.isfinite(event_input).all().item()
    assert torch.isfinite(gt_heatmap).all().item()

    assert ((event_input >= 0) & (event_input <= 1)).all().item()
    assert ((gt_heatmap >= 0) & (gt_heatmap <= 1)).all().item()

    target_count = sum(
        math.isclose(
            float(annotation["label"]),
            TARGET_LABEL,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        for annotation in sample["annotation"]
    )

    active_count = int(torch.count_nonzero(gt_heatmap).item())
    peak_count = int((gt_heatmap == 1.0).sum().item())

    if target_count == 0:
        assert active_count == 0
        assert peak_count == 0
    else:
        # 同じBinに複数の直線が入る場合を許容する。
        assert 1 <= peak_count <= target_count

        # 現在のsigma=1、5×5なら中心以外に小数値が存在する。
        assert ((gt_heatmap > 0) & (gt_heatmap < 1)).any().item()

        # 各直線の支持範囲は最大5×5。重なりや境界で減る。
        assert active_count <= target_count * KERNEL_SIZE ** 2

    # Datasetが意図した分布設定を実際に使用しているか確認
    original_h, original_w = sample["count_map"].shape[-2:]

    expected = generate_hough_target(
        annotations=sample["annotation"],
        original_w=original_w,
        original_h=original_h,
        target_w=TARGET_W,
        target_h=TARGET_H,
        R=R,
        THETA=THETA,
        target_label=TARGET_LABEL,
        sigma_rho=SIGMA_RHO,
        sigma_theta=SIGMA_THETA,
        mode=MODE,
        kernel_size=KERNEL_SIZE,
    )

    assert torch.allclose(gt_heatmap, expected), (
        "DatasetとテストのGT分布設定が一致していません。"
    )

    print(
        f"[샘플 {idx:02d}] "
        f'{sample["folder_name"]}/{sample["file_name"]} | '
        f"정답 직선={target_count}, "
        f"중심 Peak={peak_count}, "
        f"활성 Bin={active_count} | PASS"
    )

    return target_count


if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"

    batch_size = 4
    max_samples = 16

    dataset = EventDataset(root_dir, threshold=3)

    if len(dataset) == 0:
        raise RuntimeError("Dataset에 샘플이 없습니다.")

    sample_count = min(max_samples, len(dataset))

    # 하나의 시나리오에만 치우치지 않도록 전체에서 균등 선택
    indices = torch.linspace(
        0, len(dataset) - 1, steps=sample_count
    ).long().tolist()

    print(f"전체 Dataset 크기: {len(dataset)}")
    print(f"검증 샘플 수: {sample_count}")
    print(
        f"설정: {MODE}, "
        f"sigma=({SIGMA_RHO}, {SIGMA_THETA}), "
        f"kernel={KERNEL_SIZE}"
    )

    empty_count = 0

    for idx in indices:
        target_count = validate_sample(dataset[idx], idx)
        empty_count += int(target_count == 0)

    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print("\nDataLoader Batch 검증")
    loaded_count = 0

    for batch_idx, batch in enumerate(dataloader):
        B = len(batch["file_name"])

        inputs = batch["event_input"]
        targets = batch["gt_heatmap"]

        assert tuple(inputs.shape) == (B, 1, TARGET_H, TARGET_W)
        assert tuple(targets.shape) == (B, 1, R, THETA)

        assert torch.isfinite(inputs).all().item()
        assert torch.isfinite(targets).all().item()
        assert ((targets >= 0) & (targets <= 1)).all().item()

        assert len(batch["annotation"]) == B

        for local_idx in range(B):
            dataset_idx = indices[loaded_count + local_idx]
            sample = dataset[dataset_idx]

            assert torch.equal(
                inputs[local_idx], sample["event_input"]
            )
            assert torch.equal(
                targets[local_idx], sample["gt_heatmap"]
            )
            assert batch["file_name"][local_idx] == sample["file_name"]

        loaded_count += B

        print(
            f"[Batch {batch_idx}] "
            f"입력={tuple(inputs.shape)}, "
            f"GT={tuple(targets.shape)} | PASS"
        )

    assert loaded_count == sample_count

    print(f"\n검증 완료: {loaded_count}개 샘플")
    print(f"빈 GT 샘플: {empty_count}개")

    if empty_count == 0:
        print("선택된 실제 샘플에는 빈 GT 사례가 없었습니다.")

    print("Dataset → DataLoader Soft GT 연결 검증 PASS")