import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import EventDataset
from src.event_backbone import EventBackbone
from src.hough_dictionary import Multi_scale_HoughDictionary
from src.hough_decoder import HoughDecoder, Hough_fusion

if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    threshold = 3
    original_w = 224
    original_h = 224
    R = 112
    THETA = 100

    dataset = EventDataset(root_dir, threshold)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sample = next(iter(dataloader))
    event_input = sample["event_input"]

    print(f"Event_input shape : {event_input.shape}")

    model = EventBackbone()
    hough_fusion = Hough_fusion()
    decoder = HoughDecoder()

    model.eval()
    hough_fusion.eval()
    decoder.eval()

    with torch.no_grad():
        operation = model.get_coordinate_operation()

        feature_maps = model(event_input)
        feature_shapes = []

        for idx, fmap in enumerate(feature_maps):
            print(f"Feature {idx} : {fmap.shape}")
            feature_shapes.append(tuple(fmap.shape[2:]))

        hough_dictionary = Multi_scale_HoughDictionary(original_w, original_h, feature_shapes, R, THETA, operation)
        hough_feature_maps = hough_dictionary(feature_maps)
        fusion_feature = hough_fusion(hough_feature_maps)
        print(f"fusion_feature shape : {fusion_feature.shape}")
        result = decoder(fusion_feature)
        print(f"result shape : {result.shape}")
        probability_map = torch.sigmoid(result)

        # 시각화 부분
        # (B, 1, R, THETA) → 첫 샘플의 (R, THETA)
        heatmap = probability_map[0, 0].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        image = ax.imshow(
            heatmap,
            cmap="viridis",
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="auto",
        )

        ax.set_xlabel("Theta bin")
        ax.set_ylabel("Rho bin")
        ax.set_title("Predicted Hough Heatmap — Untrained Model")

        fig.colorbar(image, ax=ax, label="Sigmoid score")

        plt.tight_layout()
        plt.show()

        print(f"Min : {heatmap.min():.6f}")
        print(f"Max : {heatmap.max():.6f}")
        print(f"Mean: {heatmap.mean():.6f}")