import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.event_backbone import EventBackbone

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

    def hough_transform(self, H, W, x, y, idx):
        """
        :param H: feature map의 Height
        :param W: feature map의 Width
        :param x: feature map의 x 좌표
        :param y: feature map의 y 좌표
        :param idx : event_backbone stage index

        :return: Hough space ==> Hough dictionary로 사용
        """
        # 허프 공간 생성
        hough_space = torch.zeros((self.R, self.THETA))
        # 가상의 입력 feature map 생성
        feature_map = torch.zeros((H, W))
        feature_map[y, x] = 1

        # event count map 중심 좌표
        center_x = (self.original_w - 1) / 2
        center_y = (self.original_h - 1) / 2

        # feature map의 x,y 좌표 ---> 이전 feature map의 좌표 계산
        op = self.scale_operation[idx]
        original_x, original_y = self.previous_center(x, y, op)

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

        hough_space[rho_idx, theta_idx] += feature_map[y, x]

        return hough_space

    def create_hough_dictionary(self, H, W, idx):
        hough_dictionary = torch.zeros((H * W, self.R, self.THETA))

        HW_count = 0
        for y in range(H):
            for x in range(W):
                hough_space = self.hough_transform(H, W, x, y, idx)
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