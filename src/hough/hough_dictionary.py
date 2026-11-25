import torch
import torch.nn as nn

class Multi_scale_HoughDictionary(nn.Module):
    def __init__(self, original_w, original_h, feature_shapes, R, THETA, operation):
        super().__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.feature_shapes = feature_shapes
        self.R = R
        self.THETA = THETA
        self.scale_operation = []

        for idx, (H, W) in enumerate(self.feature_shapes):
            self.scale_operation.append(operation[:idx+2])

            dictionary = self.create_hough_dictionary(H, W, idx)
            self.register_buffer(f"dictionary_{idx}", dictionary)

    def previous_center(self, x, y, operation):
        previous_x = x
        previous_y = y

        def pair(value):
            if isinstance(value, (tuple, list)):
                return value
            return (value, value)

        for param in reversed(operation):
            kh, kw = pair(param["kernel_size"])
            sh, sw = pair(param["stride"])
            ph, pw = pair(param["padding"])

            previous_x = previous_x * sw - pw + (kw - 1) / 2
            previous_y = previous_y * sh - ph + (kh - 1) / 2

        return previous_x, previous_y

    def create_hough_dictionary(self, H, W, idx):
        hough_dictionary = torch.zeros((H * W, self.R, self.THETA))

        # feature의 모든 위치를 배열로 준비
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )

        x = x_grid.flatten()
        y = y_grid.flatten()

        original_x, original_y = self.previous_center(x, y, self.scale_operation[idx])
        # event_count_map 중심 좌표
        center_x = (self.original_w - 1) / 2
        center_y = (self.original_h - 1) / 2

        x_ = original_x - center_x
        y_ = original_y - center_y

        rho_max = torch.sqrt(
            torch.tensor(center_x ** 2 + center_y ** 2, dtype=torch.float32)
        )
        rho_step = (rho_max * 2) / self.R

        angles = torch.arange(self.THETA).float() * (180.0 / self.THETA)
        radians = torch.deg2rad(angles)

        pixel_indices = torch.arange(H*W)

        for theta_idx in range(self.THETA):
            theta = radians[theta_idx]

            rho = x_ * torch.cos(theta) + y_ * torch.sin(theta)

            rho_indices = torch.floor(
                (rho_max + rho) / rho_step
            ).long()
            rho_indices = rho_indices.clamp(0, self.R - 1)

            hough_dictionary[pixel_indices, rho_indices, theta_idx] = 1.0

        return hough_dictionary



    # multi-scale의 feature map을 해당 feature map의 해상도와 맞는 hough dictionary와 내적
    def forward(self, feature_maps):
        hough_feature_maps = []
        for idx, feature_map in enumerate(feature_maps):
            B, C, H, W = feature_map.shape
            dictionary = getattr(self, f"dictionary_{idx}")

            feature_map = feature_map.flatten(start_dim=2)
            hough_dictionary = dictionary.reshape(H*W, self.R * self.THETA)

            hough_feature_map = feature_map @ hough_dictionary
            hough_feature_map = hough_feature_map.reshape(B, C, self.R, self.THETA)

            hough_feature_maps.append(hough_feature_map)
        return hough_feature_maps