import torch
import torch.nn as nn

class Hough_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            for _ in range(4)
        ])

    def forward(self, hough_feature_maps):
        projected_features = []

        for idx, hmap in enumerate(hough_feature_maps):
            projected = self.projection[idx](hmap)
            projected_features.append(projected)

        return torch.cat(projected_features, dim=1)