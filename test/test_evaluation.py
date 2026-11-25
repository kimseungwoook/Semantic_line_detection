import torch
from torch.utils.data import DataLoader, Subset

from event_model.event_dataset import EventDataset, event_collate_fn
from event_model.event_line_model import EventLineModel
from hough.hough_postprocess import nms_2d_topk, hough_bin_to_line_points
from metric.evaluation import match_lines, precision_recall_f1, calculate_hiou

if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    torch.manual_seed(42)

    batch_size = 4
    R, Theta = 112, 100
    threshold = 3
    original_w, original_h = 224, 224

    dataset = EventDataset(root_dir, threshold)
    subset = Subset(dataset, [0, 1806, 5420, 9034])

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=event_collate_fn)
    batch = next(iter(dataloader))

    event_input = batch["event_input"].to(device)
    gt_heatmap = batch["gt_heatmap"]

    model = EventLineModel(
        original_w=original_w,
        original_h=original_h,
        R=R, Theta=Theta
    ).to(device)

    state_dict = torch.load(
        r"/model\test_event_line_model.pth",
        map_location = device,
        weights_only = True
    )

    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        probability_map = torch.sigmoid(model(event_input))

        total_tp = 0
        total_fp = 0
        total_fn = 0
        hiou_scores = []

        for idx in range(probability_map.shape[0]):
            heatmap = probability_map[idx, 0].cpu().numpy()

            peaks = nms_2d_topk(heatmap, K=6, radius=2, threshold=0.2)

            pred_lines = []

            for peak in peaks:
                line_points = hough_bin_to_line_points(
                    peak,
                    original_w=346, original_h=260,
                    input_w=224, input_h=224,
                    R=R, Theta=Theta
                )

                if line_points is not None:
                    pred_lines.append(line_points)
            gt_lines = [
                annotation["points"]
                for annotation in batch["annotation"][idx]
                if abs(annotation["label"] - 1.0) <= 1e-6
            ]

            print(
                f"Sample {idx}"
                f" | 예측 직선={len(pred_lines)}개"
                f" | GT 직선={len(gt_lines)}개"
            )

            tp, fp, fn = match_lines(
                pred_lines,
                gt_lines,
                ea_threshold=0.90,
                image_size=(260, 346)
            )

            precision, recall, f1 = precision_recall_f1(tp, fp, fn)

            hiou = calculate_hiou(
                pred_lines,
                gt_lines,
                image_size=(260, 346)
            )

            print(
                f"Sample {idx}"
                f" | TP={tp}, FP={fp}, FN={fn}"
                f" | Precision={precision:.4f}"
                f" | Recall={recall:.4f}"
                f" | F1={f1:.4f}"
                f" | HIoU={hiou:.4f}"
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            hiou_scores.append(hiou)

        precision, recall, f1 = precision_recall_f1(
            total_tp, total_fp, total_fn
        )
        mean_hiou = sum(hiou_scores) / len(hiou_scores)

        print(
            f"\n전체 | TP={total_tp}, FP={total_fp}, FN={total_fn}"
            f" | Precision={precision:.4f}"
            f" | Recall={recall:.4f}"
            f" | F1={f1:.4f}"
            f" | 평균 HIoU={mean_hiou:.4f}"
        )
