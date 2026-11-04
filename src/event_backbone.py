import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from src.dataset import EventDataset

class EventBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.stage1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3

    def forward(self, x):
        feature_map1 = self.stage1(x)
        feature_map2 = self.stage2(feature_map1)
        feature_map3 = self.stage3(feature_map2)

        return [feature_map1, feature_map2, feature_map3]