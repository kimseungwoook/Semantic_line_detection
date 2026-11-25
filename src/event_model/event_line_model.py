import torch
import torch.nn as nn

from src.hough_dictionary import Multi_scale_HoughDictionary
from src.event_backbone import EventBackbone, FeatureProjection
from src.hough_decoder import HoughDecoder
from src.hough_fusion import Hough_Fusion

class EventLineModel(nn.Module):
    def __init__(self, original_w=224, original_h=224, R=112, Theta=100, dropout_p=0.0):
        super().__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.R = R
        self.Theta = Theta
        self.feature_shapes = [(56, 56), (28, 28), (14, 14), (7,7)]
        self.event_backbone = EventBackbone()
        self.feature_projection = FeatureProjection()
        self.operation = self.event_backbone.get_coordinate_operation()

        self.hough_dictionary = Multi_scale_HoughDictionary(
            original_w = self.original_w,
            original_h = self.original_h,
            feature_shapes = self.feature_shapes,
            R = self.R,
            THETA = self.Theta,
            operation = self.operation
        )
        self.hough_fusion = Hough_Fusion()
        self.HoughDecoder = HoughDecoder(dropout_p=dropout_p)

    def forward(self, x):
        feature_maps = self.event_backbone(x)
        feature_maps = self.feature_projection(feature_maps)

        hough_feature_maps = self.hough_dictionary(feature_maps)

        fusion_feature = self.hough_fusion(hough_feature_maps)

        hough_space_heatmap = self.HoughDecoder(fusion_feature)

        return hough_space_heatmap
