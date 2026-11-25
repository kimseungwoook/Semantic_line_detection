import torch.nn as nn

from src.rgb_model.rgb_backbone import RGBbackbone
from src.hough.hough_dictionary import Multi_scale_HoughDictionary
from src.hough.hough_fusion import Hough_Fusion
from src.Decoder import HoughDecoder
from src.feature_projection import FeatureProjection

class Rgb_line_model(nn.Module):
    def __init__(self, original_w=224, original_h=224, R=112, Theta=100, dropout_p=0.0, imagenet_pretrained=False):
        super().__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.R = R
        self.Theta = Theta
        self.feature_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
        self.rgb_backbone = RGBbackbone(imagenet_pretrained)
        self.feature_projection = FeatureProjection()
        self.operation = self.rgb_backbone.get_coordinate_operation()

        self.hough_dictionary = Multi_scale_HoughDictionary(
            original_w=self.original_w,
            original_h=self.original_h,
            R=self.R,
            THETA=self.Theta,
            operation=self.operation,
            feature_shapes=self.feature_shapes
        )
        self.hough_fusion = Hough_Fusion()
        self.HoughDecoder = HoughDecoder(dropout_p)

    def forward(self, x):
        feature_maps = self.rgb_backbone(x)
        feature_maps = self.feature_projection(feature_maps)

        hough_feature_maps = self.hough_dictionary(feature_maps)
        fusion_feature_maps = self.hough_fusion(hough_feature_maps)

        hough_space_heatmap = self.HoughDecoder(fusion_feature_maps)

        return hough_space_heatmap