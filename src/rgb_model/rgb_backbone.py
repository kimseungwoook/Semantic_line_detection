import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class RGBbackbone(nn.Module):
    def __init__(self, imagenet_pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        backbone = resnet50(weights=weights)
        self.stage1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1
        )
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        feature_map1 = self.stage1(x)
        feature_map2 = self.stage2(feature_map1)
        feature_map3 = self.stage3(feature_map2)
        feature_map4 = self.stage4(feature_map3)

        return [feature_map1, feature_map2, feature_map3, feature_map4]

    def get_coordinate_operation(self):
        operation = []

        def add_operation(layer):
            operation.append({
                "kernel_size" : layer.kernel_size,
                "stride" : layer.stride,
                "padding" : layer.padding
            })

        add_operation(self.stage1[0])
        add_operation(self.stage1[3])
        add_operation(self.stage2[0].conv2)
        add_operation(self.stage3[0].conv2)
        add_operation(self.stage4[0].conv2)

        return operation