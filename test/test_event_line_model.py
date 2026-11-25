import torch
from torch.utils.data import DataLoader, Subset

from event_model.event_dataset import EventDataset, event_collate_fn
from event_model.event_line_model import EventLineModel
from metric.loss import FocalLoss

def check_tensor(name, tensor, expected_shape):
    assert tuple(tensor.shape) == expected_shape, (
        f"{name}: 예상 {expected_shape}, 실제 {tuple(tensor.shape)}"
    )
    assert torch.isfinite(tensor).all().item(), (
        f"{name}: NaN 또는 Inf 발생"
    )
    print(f"{name}: {tuple(tensor.shape)}, PASS")

if __name__ == "__main__":
    torch.manual_seed(0)

    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    batch_size = 4
    R, Theta = 112, 100
    threshold = 3
    original_w, original_h = 224, 224

    dataset = EventDataset(root_dir, threshold)
    subset = Subset(dataset, [0, 1806, 5420, 9034])

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=event_collate_fn,
    )

    batch = next(iter(dataloader))
    event_input = batch["event_input"].to(device)
    gt_heatmap = batch["gt_heatmap"].to(device)

    model = EventLineModel(
        original_w = original_w,
        original_h = original_h,
        R = R, Theta = Theta
    ).to(device)

    check_tensor("입력", event_input, (batch_size, 1, 224, 224))
    check_tensor("GT", gt_heatmap, (batch_size, 1, R, Theta))

    assert ((gt_heatmap >= 0) & (gt_heatmap <= 1)).all().item()

    # 평가 모드에서 단계별 shape 확인
    model.eval()

    with torch.no_grad():
        features = model.event_backbone(event_input)
        expected_shapes = [
            (batch_size, 256, 56, 56),
            (batch_size, 512, 28, 28),
            (batch_size, 1024, 14, 14),
            (batch_size, 2048, 7, 7)
        ]

        assert len(features) == len(expected_shapes)

        # Backbone feature map Shape 검사
        for idx, feature in enumerate(features):
            check_tensor(f"Backbone {idx}", feature, expected_shapes[idx])

        # 여러 Scale feature map 1x1 conv : feature->128 검사
        projected_features = model.feature_projection(features)
        for idx, feature in enumerate(projected_features):
            height, width = expected_shapes[idx][-2:]
            check_tensor(
                f"FeatureProjection {idx}",
                feature,
                (batch_size, 128, height, width)
            )

        # 허프 feature map shape 검사 (batch_size, 128, R, Theta)
        hough_feature_maps = model.hough_dictionary(projected_features)
        for idx, feature in enumerate(hough_feature_maps):
            check_tensor(
                f"Hough feature map {idx}",
                feature, (batch_size, 128, R, Theta)
            )

        # Hough feature map 1x1 conv (128 -> 64) ----> concat -> (batch_size, 256, R, Theta)
        hough_fusion = model.hough_fusion(hough_feature_maps)
        check_tensor(
            "Fusion", hough_fusion, (batch_size, 256, R, Theta)
        )

        # Hough fusion을 Hough Decoder의 입력 값으로 사용
        # Decoder 최종 출력 : (Batch_size, 1, R, Theta)
        output = model.HoughDecoder(hough_fusion)
        check_tensor(
            "Hough Decoder output", output, (batch_size, 1, R, Theta)
        )

        # 통합 forward가 같은 결과를 만드는지 확인
        integrated_output = model(event_input)
        assert torch.allclose(integrated_output, output)
        print("EventLineModel 연결 PASS")

    # 2. 학습 모드에서 Loss와 역전파 확인
    model.train()
    model.zero_grad(set_to_none=True)

    prediction = model(event_input)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    loss = criterion(prediction, gt_heatmap)

    assert torch.isfinite(loss).item()
    print(f"\nFocal Loss: {loss.item():.6f}")

    loss.backward()

    modules = {
        "Backbone" : model.event_backbone,
        "FeatureProjection" : model.feature_projection,
        "Hough_Fusion" : model.hough_fusion,
        "Hough_Decoder" : model.HoughDecoder
    }

    for name, module in modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert gradients, f"{name}: 학습 파라미터 없음"
        assert all(gradient is not None for gradient in gradients), (
            f"{name}: gradient가 없는 파라미터 존재"
        )
        assert all(
            torch.isfinite(gradient).all().item()
            for gradient in gradients
        ), f"{name}: gradient에 NaN 또는 Inf 발생"

        assert any(
            gradient.abs().max().item() > 0
            for gradient in gradients
        ), f"{name}: 모든 gradient가 0"

        print(f"{name}: Gradient PASS")

    print("\n4-Scale 모델 Shape·유한값·Loss·역전파 검증 PASS")

