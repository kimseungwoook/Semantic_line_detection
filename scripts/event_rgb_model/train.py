import torch
import argparse
import csv
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from src.rgb_event_dataset import Rgb_Event_Dataset, rgb_event_collate_fn
from src.rgb_event_line_model import rgb_event_line_model
from src.metric.loss import FocalLoss
from src.hough.hough_postprocess import nms_2d_topk, hough_bin_to_line_points
from src.metric.evaluation import match_lines, precision_recall_f1, calculate_hiou
from scripts.plot_training import save_training_plot

if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Train the Rgb_Event line model")
    parser.add_argument("--model_name", type=str, default="Rgb_Event_line_model")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--event_clip", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--nms_radius", type=int, default=2)
    parser.add_argument("--score_threshold", type=float, default=0.2)
    parser.add_argument("--ea_threshold", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=0.001)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {args.gpu}를 사용할 수 없습니다.")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    print(f"-----Parameter-----\n"
          f"gpu : {args.gpu}\n"
          f"model_name : {args.model_name}\n"
          f"dropout : {args.dropout}\n"
          f"imagenet_pretrained : {args.imagenet_pretrained}"
          f"epochs : {args.epochs}\n"
          f"batch_size : {args.batch_size}\n"
          f"lr : {args.lr}\n"
          f"weight_decay : {args.weight_decay}\n"
          f"event_clip : {args.event_clip}\n"
          f"top_k : {args.top_k}\n"
          f"nms_radius : {args.nms_radius}\n"
          f"score_threshold : {args.score_threshold}\n"
          f"ea_threshold : {args.ea_threshold}\n"
          f"patience : {args.patience}\n"
          f"min_delta : {args.min_delta}\n"
          f"-------------------")

    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    project_dir = Path(__file__).resolve().parents[2]

    save_dir = project_dir / "model" / "rgb_event_model" / f"{args.model_name}_ea_{args.ea_threshold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.model_name}_best_hiou.pth"
    history_path = save_dir / "history.csv"
    split_path = project_dir / "splits" / "split.csv"

    dataset = Rgb_Event_Dataset(root_dir, event_clip=args.event_clip)

    folder_to_split = {}
    with split_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            folder_to_split[row["folder_name"]] = row["split"]

    split_indices = {
        "train": [],
        "val": [],
        "test": []
    }

    for idx, (_, _, folder_name, _) in enumerate(dataset.pairs):
        split_indices[folder_to_split[folder_name]].append(idx)

    train_dataset = Subset(dataset, split_indices["train"])
    val_dataset = Subset(dataset, split_indices["val"])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=rgb_event_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=rgb_event_collate_fn
    )

    model = rgb_event_line_model(
        dropout_p=args.dropout,
        image_trained=args.imagenet_pretrained
    ).to(device)

    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_hiou = -1.0
    stopping_reference_hiou = float("-inf")
    epochs_without_improvement = 0

    history = []
    fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall", "f1", "hiou"]
    with history_path.open("x", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    for epoch in range(args.epochs):
        model.train()

        train_loss_sum = 0.0
        train_seen = 0
        train_bar = tqdm(train_dataloader, desc=f"Train {epoch + 1}/{args.epochs}", leave=False)

        for batch in train_bar:
            optimizer.zero_grad()

            rgb_input = batch["rgb_input"].to(device)
            event_input = batch["event_input"].to(device)
            gt_heatmap = batch["gt_heatmap"].to(device)

            prediction = model(event_input=event_input, rgb_input=rgb_input)

            loss = criterion(prediction, gt_heatmap)
            loss.backward()

            optimizer.step()

            batch_size = rgb_input.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_seen += batch_size
            train_bar.set_postfix(loss=f"{train_loss_sum / train_seen:.6f}")

        train_loss_mean = train_loss_sum / len(train_dataset)

        model.eval()

        val_loss_sum = 0.0
        val_seen = 0
        val_bar = tqdm(val_dataloader, desc=f"Val {epoch + 1}/{args.epochs}", leave=False)

        total_tp, total_fp, total_fn = 0, 0, 0
        hiou_scores = []
        with torch.no_grad():
            for batch in val_bar:
                rgb_input = batch["rgb_input"].to(device)
                event_input = batch["event_input"].to(device)
                gt_heatmap = batch["gt_heatmap"].to(device)
                annotation = batch["annotation"]

                prediction = model(event_input=event_input, rgb_input=rgb_input)
                loss = criterion(prediction, gt_heatmap)
                probability_map = torch.sigmoid(prediction)

                for idx in range(probability_map.shape[0]):
                    heatmap = probability_map[idx].squeeze(0).cpu().numpy()

                    peaks = nms_2d_topk(
                        heatmap=heatmap, K=args.top_k,
                        radius=args.nms_radius,
                        threshold=args.score_threshold
                    )

                    pred_lines = []
                    for peak in peaks:
                        line = hough_bin_to_line_points(peak)

                        if line is not None:
                            pred_lines.append(line)

                    gt_lines = [
                        anno["points"]
                        for anno in annotation[idx]
                        if abs(anno["label"] - 1.0) <= 1e-6
                    ]

                    tp, fp, fn = match_lines(
                        pred_lines, gt_lines,
                        ea_threshold=args.ea_threshold
                    )
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                    hiou = calculate_hiou(pred_lines, gt_lines)
                    hiou_scores.append(hiou)

                batch_size = rgb_input.shape[0]
                val_loss_sum += loss.item() * batch_size
                val_seen += batch_size
                val_bar.set_postfix(loss=f"{val_loss_sum / val_seen:.6f}")

            precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)

        val_loss_mean = val_loss_sum / len(val_dataset)
        mean_hiou = sum(hiou_scores) / len(hiou_scores)
        if mean_hiou > best_hiou:
            best_hiou = mean_hiou

            if isinstance(model, torch.nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model

            torch.save(model_to_save.state_dict(), save_path)

            print(
                f"모델 저장 | Epoch={epoch + 1}"
                f" | Best HIoU={best_hiou:.4f}"
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_mean,
            "val_loss": val_loss_mean,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hiou": mean_hiou,
        })
        with history_path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writerow(history[-1])

        print(
            f"Epoch {epoch + 1}/{args.epochs}",
            f" | Train Loss : {train_loss_mean:.6f}",
            f" | Val Loss : {val_loss_mean:.6f}"
            f" | P={precision:.4f}",
            f" | R={recall:.4f}",
            f" | F1={f1:.4f}",
            f" | HIoU={mean_hiou:.4f}"
        )

        if mean_hiou >= stopping_reference_hiou + args.min_delta:
            stopping_reference_hiou = mean_hiou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping | Epoch={epoch + 1}"
                f" | Best Val HIoU={best_hiou:.4f}"
                f" | No significant improvement for {epochs_without_improvement} epochs"
            )
            break

    plot_path = save_dir / "learning_curves.png"
    save_training_plot(history, plot_path, args.ea_threshold)
    print(f"학습 그래프 저장: {plot_path}")