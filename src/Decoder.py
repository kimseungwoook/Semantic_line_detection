import torch
import torch.nn as nn

class HoughDecoder(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
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
        self.stage = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=1,
                kernel_size=1, stride=1, padding=0
            )
        )

    def forward(self, x):
        feature1 = self.branch1(x)
        feature2 = self.branch2(x)
        feature3 = self.branch3(x)

        x = torch.cat([feature1, feature2, feature3], dim=1)
        x = self.dropout(x)
        x = self.stage(x)
        return x