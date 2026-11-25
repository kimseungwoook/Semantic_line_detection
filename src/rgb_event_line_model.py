import torch
import torch.nn as nn

from src.event_model.event_backbone import EventBackbone
from src.rgb_model.rgb_backbone import RGBbackbone
from src.feature_projection import FeatureProjection
from src.hough.hough_dictionary import Multi_scale_HoughDictionary
from src.hough.hough_fusion import Hough_Fusion
from src.Decoder import HoughDecoder

class rgb_event_line_model(nn.Module):
    def __init__(self, original_w=224, original_h=224, R=112, Theta=100, dropout_p=0.0, image_trained=False):
        super().__init__()
        self.feature_shape = [(56, 56), (28, 28), (14, 14), (7,7)]

        self.ev_backbone = EventBackbone(image_trained)
        self.ev_operation = self.ev_backbone.get_coordinate_operation()
        self.rgb_backbone = RGBbackbone(image_trained)
        self.rgb_operation = self.rgb_backbone.get_coordinate_operation()

        self.ev_hough_dictionary = Multi_scale_HoughDictionary(
            original_w = original_w,
            original_h = original_h,
            R = R, THETA = Theta,
            feature_shapes = self.feature_shape,
            operation = self.ev_operation
        )
        self.rgb_hough_dictionary = Multi_scale_HoughDictionary(
            original_w = original_w,
            original_h = original_h,
            R = R, THETA = Theta,
            feature_shapes = self.feature_shape,
            operation = self.rgb_operation
        )

        self.ev_projection = FeatureProjection()
        self.rgb_projection = FeatureProjection()

        self.ev_hough_fusion = Hough_Fusion()
        self.rgb_hough_fusion = Hough_Fusion()

        self.modal_fusion = nn.Conv2d(512, 256, kernel_size=1)
        self.decoder = HoughDecoder(dropout_p=dropout_p)

    def forward(self, event_input, rgb_input):
        ev_features = self.ev_projection(self.ev_backbone(event_input))
        rgb_features = self.rgb_projection(self.rgb_backbone(rgb_input))

        ev_hough = self.ev_hough_fusion(self.ev_hough_dictionary(ev_features))
        rgb_hough = self.rgb_hough_fusion(self.rgb_hough_dictionary(rgb_features))

        combined = torch.cat([ev_hough, rgb_hough], dim=1)
        fused = self.modal_fusion(combined)
        hough_space = self.decoder(fused)

        return hough_space