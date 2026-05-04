import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from ICM_Module import ICM


class TSIFM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ir_embed = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        ])
        self.vi_embed = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        ])

        self.ir_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.vi_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.ir_self_improve = ICM(dim=out_channels, reduction=reduction)
        self.vi_self_improve = ICM(dim=out_channels, reduction=reduction)

        self.ir_scale_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.vi_scale_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.residual_ir = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual_vi = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1.)
                constant_(m.bias, 0.)

    def forward(self, x, y):
        residual_x = self.residual_ir(x)
        residual_y = self.residual_vi(y)

        ir_small = self.ir_embed[0](x)
        ir_large = self.ir_embed[1](x)
        vi_small = self.vi_embed[0](y)
        vi_large = self.vi_embed[1](y)

        ir_scale = self.ir_scale_fuse(torch.cat([ir_small, ir_large], dim=1))
        vi_scale = self.vi_scale_fuse(torch.cat([vi_small, vi_large], dim=1))

        ir_gate_weight = self.ir_gate(torch.cat([ir_scale, vi_scale], dim=1))
        ir_cross = ir_scale * ir_gate_weight + vi_scale * (1 - ir_gate_weight)
        ir_cross = ir_cross + 0.1 * ir_scale

        vi_gate_weight = self.vi_gate(torch.cat([vi_scale, ir_scale], dim=1))
        vi_cross = vi_scale * vi_gate_weight + ir_scale * (1 - vi_gate_weight)
        vi_cross = vi_cross + 0.1 * vi_scale

        ir_self = self.ir_self_improve([ir_cross, ir_scale])
        ir_self = ir_self + ir_cross

        vi_self = self.vi_self_improve([vi_cross, vi_scale])
        vi_self = vi_self + vi_cross

        out_x = F.layer_norm(ir_self + residual_x, normalized_shape=ir_self.shape[1:])
        out_y = F.layer_norm(vi_self + residual_y, normalized_shape=vi_self.shape[1:])

        return out_x, out_y
