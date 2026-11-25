import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.rgb_model.rgb_dataset import Rgb_Dataset, rgb_collate_fn
from src.rgb_model.rgb_line_model import Rgb_line_model
from src.metric.loss import FocalLoss

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    dataset = Rgb_Dataset(root_dir)
    subset = Subset(dataset, [0, 1806, 5420, 9034])

    original_w, original_h = 224, 224
    R = 112
    Theta = 100

    dataloader = DataLoader(
        subset, shuffle=False,
        batch_size=4,num_workers=0,
        collate_fn = rgb_collate_fn
    )

    batch = next(iter(dataloader))

    rgb_input = batch["rgb_input"].to(device)
    gt_heatmap = batch["gt_heatmap"].to(device)

    model = Rgb_line_model(
        original_h=original_h,
        original_w=original_w,
        R=R, Theta=Theta
    ).to(device)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    steps = 1000
    for i in range(steps):
        optimizer.zero_grad()

        prediction = model(rgb_input)

        loss = criterion(prediction, gt_heatmap)
        loss.backward()

        optimizer.step()
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{steps} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        probability_map = torch.sigmoid(model(rgb_input))
        gt_images = gt_heatmap.detach().cpu()
        pred_images = probability_map.detach().cpu()

        fig, axes = plt.subplots(
            4, 2,
            figsize=(10, 3 * 4),
            squeeze=False,
            constrained_layout=True
        )

        for idx in range(4):
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