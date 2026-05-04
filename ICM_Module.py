import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

class ICM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(ICM, self).__init__()
        self.height = height
        self.dim = dim
        d = max(int(dim / reduction), 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, dim * height, 1, bias=False),
            nn.BatchNorm2d(dim * height)
        )
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1.)
                constant_(m.bias, 0.)

    def forward(self, in_feats):
        feat1, feat2 = in_feats
        B, C, H, W = feat1.shape
        feats_sum = feat1 + feat2
        avg_feat = self.avg_pool(feats_sum)
        max_feat = self.max_pool(feats_sum)
        global_feat = torch.cat([avg_feat, max_feat], dim=1)
        attn = self.mlp(global_feat)
        attn = attn.view(B, self.height, C, 1, 1)
        attn = self.softmax(attn)
        in_feats_stacked = torch.stack(in_feats, dim=1)
        out = torch.sum(in_feats_stacked * attn, dim=1)
        return out
