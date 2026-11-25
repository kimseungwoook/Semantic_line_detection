import torch.nn as nn

class FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fmap1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fmap2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fmap3 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fmap4 = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, feature_maps):
        feature_map1 = self.fmap1(feature_maps[0])
        feature_map2 = self.fmap2(feature_maps[1])
        feature_map3 = self.fmap3(feature_maps[2])
        feature_map4 = self.fmap4(feature_maps[3])

        return [feature_map1, feature_map2, feature_map3, feature_map4]