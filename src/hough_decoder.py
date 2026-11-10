import torch
import torch.nn as nn


# Hough_feature map Scale별 채널 축소 -> concatenate
class Hough_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Conv2d(1024, 128, kernel_size=1)
        ])

    def forward(self, hough_feature_maps):
        projected_features = []
        for idx, hmap in enumerate(hough_feature_maps):
            projected = self.projection[idx](hmap)
            projected_features.append(projected)

        fusion_feature = torch.cat(projected_features, dim=1)

        return fusion_feature

class HoughDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(384, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=64,
                kernel_size=3, stride=1, padding=1,
                dilation=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=64,
                kernel_size=3, stride=1, padding=2,
                dilation=2, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=64,
                kernel_size=3, stride=1, padding=3,
                dilation=3, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=1,
                kernel_size=1, stride=1, padding=0
            )
        )

    def forward(self, x):
        x = self.stage1(x)

        feature1 = self.branch1(x)
        feature2 = self.branch2(x)
        feature3 = self.branch3(x)

        x = torch.cat([feature1, feature2, feature3], dim=1)

        x = self.stage2(x)
        return x