import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import kaiming_normal_, constant_

class Denoise(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
    def _create_gaussian_kernel(self, kernel_size, sigma):
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1).float()
        y = torch.arange(-radius, radius + 1).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, Amp):
        b, c, h, w = Amp.shape
        denoised_Amp = []
        for i in range(b):
            single_Amp = Amp[i:i + 1]
            single_denoised = F.conv2d(
                single_Amp,
                self.kernel,
                padding=self.kernel_size // 2,
                groups=1
            )
            denoised_Amp.append(single_denoised)
        denoised_Amp = torch.cat(denoised_Amp, dim=0)
        return denoised_Amp

class AFDFM (nn.Module):
    def __init__(self):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )
        self.denoise = Denoise(kernel_size=3, sigma=0.8)
        self.cutoff_ratio = nn.Parameter(torch.tensor(0.25))
    def create_freq_mask(self, h, w, device):
        y = torch.arange(h, device=device).float() - h // 2
        x = torch.arange(w, device=device).float() - w // 2
        y, x = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(y ** 2 + x ** 2) / max(h // 2, w // 2)
        low_mask = torch.sigmoid((self.cutoff_ratio - dist) / 0.05).unsqueeze(0).unsqueeze(0)
        high_mask = 1 - low_mask
        return low_mask, high_mask

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        low_mask, high_mask = self.create_freq_mask(H, W, f1.device)
        low_x = torch.cat([f1 * low_mask, f2 * low_mask], dim=1)
        low_fused = self.low_conv(low_x)
        f1_denoised = self.denoise(f1)
        f2_denoised = self.denoise(f2)
        high_x = torch.cat([f1_denoised * high_mask, f2_denoised * high_mask], dim=1)
        high_fused = self.high_conv(high_x)
        weight = self.weight_gen(torch.cat([f1, f2], dim=1))
        fused = weight * low_fused + (1 - weight) * high_fused
        return fused