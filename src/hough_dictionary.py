import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Multi_scale_HoughDictionary(nn.Module):
    def __init__(self, original_w, original_h, feature_shapes, R, THETA):
        super().__init__()
        self.original_w = original_w
        self.original_h = original_h
        self.feature_shapes = feature_shapes
        self.R = R
        self.THETA = THETA

        for idx, (H, W) in enumerate(self.feature_shapes):
            dictionary = self.create_hough_dictionary(H, W)
            self.register_buffer(f"dictionary_{idx}", dictionary)

    def hough_transform(self, H, W, x, y):
        # 허프 공간 생성
        hough_space = torch.zeros((self.R, self.THETA))
        # 입력 event_count_map 생성
        event_count_map = torch.zeros((H, W))
        event_count_map[y, x] = 1

        center_x = (self.original_w - 1) / 2
        center_y = (self.original_h - 1) / 2

        # 원점을 이미지 중심좌표로 하였을 때 좌표
        original_x = x * (self.original_w - 1) / (W - 1)
        original_y = y * (self.original_h - 1) / (H - 1)

        x_ = original_x - center_x
        y_ = original_y - center_y

        # rho_max : 최대 직선 거리
        value = center_x ** 2 + center_y ** 2
        rho_max = torch.sqrt(torch.tensor(value, dtype=torch.float32))
        rho_step = (rho_max * 2) / self.R

        theta_idx = torch.arange(self.THETA)
        angle = theta_idx.float() * (180.0 / self.THETA)
        rad = torch.deg2rad(angle)

        # 허프 곡선 계산
        rho = x_ * torch.cos(rad) + y_ * torch.sin(rad)

        # 실제 거리를 허프 공간의 행 번호로 변환
        rho_idx = torch.floor((rho_max + rho) / rho_step).long()
        rho_idx = torch.clamp(rho_idx, 0, self.R - 1)

        hough_space[rho_idx, theta_idx] += event_count_map[y, x]

        return hough_space

    def create_hough_dictionary(self, H, W):
        hough_dictionary = torch.zeros((H * W, self.R, self.THETA))

        HW_count = 0
        for y in range(H):
            for x in range(W):
                hough_space = self.hough_transform(H, W, x, y)
                hough_dictionary[HW_count] = hough_space
                HW_count += 1

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
