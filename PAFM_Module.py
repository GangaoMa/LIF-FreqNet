import torch
import torch.nn as nn
import torch.nn.functional as F

class PAFM(nn.Module):
    def __init__(self):
        super().__init__()

        self.offset_estimator = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 2, kernel_size=1, bias=False),
            nn.Tanh()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def _warp_phase(self, phase, offset):
        B, C, H, W = phase.shape

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_y = grid_y.float().to(phase.device) / (H - 1) * 2 - 1
        grid_x = grid_x.float().to(phase.device) / (W - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)

        offset = offset.permute(0, 2, 3, 1)
        grid = grid + offset * 0.05

        warped_phase = F.grid_sample(
            phase, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return warped_phase

    def forward(self, f1, f2):

        phase_cat = torch.cat([f1, f2], dim=1)

        offset = self.offset_estimator(phase_cat)
        ir_pha_aligned = self._warp_phase(f1, offset)

        x = torch.cat([ir_pha_aligned, f2], dim=1)
        x = self.conv1(x)
        fused_phase = torch.tanh(x)

        return fused_phase