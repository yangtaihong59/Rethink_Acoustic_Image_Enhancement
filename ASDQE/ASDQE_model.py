# Copyright 2025 Taihong Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        # 如果使用双线性插值，使用普通卷积减少通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 convolution, for final output"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """Simplified UNet, for feature enhancement"""
    def __init__(self, inp_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.n_channels = inp_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # 下采样路径
        self.inc = DoubleConv(inp_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor) 
        
        # 上采样路径
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)           # [B, 64, H, W]
        x2 = self.down1(x1)        # [B, 128, H/2, W/2]
        x3 = self.down2(x2)        # [B, 256, H/4, W/4]
        x4 = self.down3(x3)        # [B, 256, H/8, W/8]
        
        x = self.up1(x4, x3)       # [B, 128, H/4, W/4]
        x = self.up2(x, x2)        # [B, 64, H/2, W/2]
        x = self.up3(x, x1)        # [B, 64, H, W]
        
        logits = self.outc(x)      # [B, out_channels, H, W]
        return logits

def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pad the height and width of the 4D Tensor (B,C,H,W) to the multiple of multiple"""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x

class DenoiseRatePredictor(nn.Module):
    """
    Denoising rate prediction model: By comparing low-quality images (LQ) and high-quality images (GT), predict the denoising rate of the image (0~1)
    """
    def __init__(self, in_channels: int = 3, dim: int = 16) -> None:
        super().__init__()
        self.unet_multiple = dim
        mid_ch = dim

        self.lq_extractor = DoubleConv(in_channels, mid_ch)
        
        self.gt_extractor = DoubleConv(in_channels, mid_ch)
        
        self.diff_extractor = DoubleConv(in_channels, mid_ch)
        
        self.unet = UNet(
            inp_channels=mid_ch * 3,  
            out_channels=mid_ch * 3
        )
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mid_ch * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.regressor[-2].bias.data.fill_(0.0)

    def forward(self, lq: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        lq = pad_to_multiple(lq, self.unet_multiple)
        gt = pad_to_multiple(gt, self.unet_multiple)
        
        lq_feat = self.lq_extractor(lq)
        gt_feat = self.gt_extractor(gt)
        diff_feat = self.diff_extractor(lq - gt)  # 差异特征

        merged_feat = torch.cat([lq_feat, gt_feat, diff_feat], dim=1)
        enhanced_feat = self.unet(merged_feat)
        score = self.regressor(enhanced_feat)
        # denoise_rate = (raw_output + 1) / 2
        
        return score