import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Subset, DataLoader

from src.dataset import EventDataset, event_collate_fn
from src.event_line_model import EventLineModel
from src.loss import FocalLoss

if __name__ == "__main__":
    torch.manual_seed(42)
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
        subset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=event_collate_fn
    )

    batch = next(iter(dataloader))
    event_input = batch["event_input"].to(device)
    gt_heatmap = batch["gt_heatmap"].to(device)

    model = EventLineModel(
        original_h=original_h, original_w=original_w,
        R=R, Theta=Theta
    ).to(device)

    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    steps = 1000
    model.train()
    for i in range(steps):
        optimizer.zero_grad()

        prediction = model(event_input)

        loss = criterion(prediction, gt_heatmap)
        loss.backward()

        optimizer.step()
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{steps} | Loss: {loss.item():.6f}")

    checkpoint_path = Path("checkpoints") / "test_event_line_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"모델 저장: {checkpoint_path.resolve()}")

    model.eval()
    with torch.no_grad():
        probability_map = torch.sigmoid(model(event_input))
        gt_images = gt_heatmap.detach().cpu()
        pred_images = probability_map.detach().cpu()

        fig, axes = plt.subplots(
            batch_size, 2,
            figsize=(10, 3 * batch_size),
            squeeze=False,
            constrained_layout=True
        )

        for idx in range(batch_size):
            axes[idx, 0].imshow(
                gt_images[idx, 0],
                origin="lower",
                cmap="viridis",
                vmin=0, vmax=1
            )
            axes[idx, 0].set_title(f"Sample {idx}: GT")

            image = axes[idx, 1].imshow(
                pred_images[idx, 0],
                origin="lower",
                cmap="viridis",
                vmin=0, vmax=1
            )
            axes[idx, 1].set_title(f"Sample {idx}: Eval prediction")

            for ax in axes[idx]:
                ax.set_xlabel("Theta bin")
                ax.set_ylabel("Rho bin")

        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Heatmap value"
        )

        plt.show()
