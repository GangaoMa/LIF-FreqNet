import torch
import torch.nn as nn
import time
import os
from thop import profile
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from AFDFM_Module import AFDFM
from TSIFM_Module import TSIFM
from PAFM_Module import PAFM

def fft(input):
    img_fft = torch.fft.rfftn(input, dim=(-2, -1))
    amp = torch.abs(img_fft)
    pha = torch.angle(img_fft)
    return amp, pha


class TSAFusionBlock(nn.Module):
    def __init__(self, dim, channels=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(8 * 3, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1)
        )

        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8 * 3, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, ir, vi, frefus):
        x = torch.cat([ir, vi, frefus], dim=1)

        w = self.weight_net(x)
        w1, w2, w3 = torch.chunk(w, 3, dim=1)

        fused = ir * w1 + vi * w2 + frefus * w3

        fused_mean = torch.mean(fused, dim=1, keepdim=True)

        out = self.conv(x) + fused_mean
        return out


class IFFT(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, 3, 1, 1, bias=False),
            nn.GELU(),  
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1, bias=False),
            nn.GELU()
        )

    def forward(self, amp, pha):

        real = amp * torch.cos(pha) + 1e-8
        imag = amp * torch.sin(pha) + 1e-8
        x = torch.complex(real, imag)
        x = torch.abs(torch.fft.irfftn(x, dim=(-2, -1)))  # [B,1,H,W]

        x = self.pre_conv(x) + x

        x_max = torch.max(x, 1, keepdim=True)[0]
        x_mean = torch.mean(x, 1, keepdim=True)
        x = torch.cat([x_max, x_mean], dim=1)

        x = self.conv1(x)
        return x

class Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 8
        self.Tsifm = TSIFM(1, self.channel)
        self.ff1 = AFDFM()
        self.ff2 = PAFM()
        self.ifft = IFFT(self.channel)
        self.fus_block = TSAFusionBlock(self.channel * 3)

        self.freq_weight = nn.Parameter(torch.tensor(0.5))
        self.spatial_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, ir, vi, epoch=0):
        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)
        amp = self.ff1(ir_amp, vi_amp)
        pha = self.ff2(ir_pha, vi_pha)
        frefus = self.ifft(amp, pha)

        frefus = self.freq_weight * frefus + (1 - self.freq_weight) * ir
        ir_feat, vi_feat = self.Tsifm(ir, vi)

        ir_feat = self.spatial_weight * ir_feat + (1 - self.spatial_weight) * ir
        vi_feat = self.spatial_weight * vi_feat + (1 - self.spatial_weight) * vi

        fus = self.fus_block(ir_feat, vi_feat, frefus)
        fus = fus + 0.03 * ir + 0.03 * vi
        fus = (fus - torch.min(fus)) / (torch.max(fus) - torch.min(fus) + 1e-8)
        return fus, amp, pha


if __name__ == "__main__":
    data_root = r"D:\python\pycharm\pytorchlearn\LIF-FreqNet\LIF-FreqNet\MSRS-main\MSRS-main\test"
    ir_dir = os.path.join(data_root, "ir")
    vi_dir = os.path.join(data_root, "vi")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use the device: {device}")

    model = Fuse().to(device).eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    print(f"Parameters: {params_m:.3f} M")

    h, w = 480, 640
    ir_dummy = torch.randn(1, 1, h, w).to(device)
    vi_dummy = torch.randn(1, 1, h, w).to(device)
    flops, _ = profile(model, inputs=(ir_dummy, vi_dummy), verbose=False)
    flops_g = flops / 1e9
    print(f"FLOPs: {flops_g:.3f} G")

    transform = transforms.ToTensor()
    ir_files = sorted(os.listdir(ir_dir))
    total_time = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(20):
            model(ir_dummy, vi_dummy)

    with torch.no_grad():
        print(f"Start testing the inference time of {len(ir_files)} images...")
        for name in ir_files:
            ir_path = os.path.join(ir_dir, name)
            vi_path = os.path.join(vi_dir, name)
            if not os.path.exists(vi_path):
                continue

            ir_img = transform(Image.open(ir_path).convert('L')).unsqueeze(0).to(device)
            vi_img = transform(Image.open(vi_path).convert('L')).unsqueeze(0).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            model(ir_img, vi_img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            count += 1

    avg_time = total_time / count

    print("\n" + "=" * 50)
    print(f"    Model Performance Metrics (MSRS Test Set)")
    print("=" * 50)
    print(f"Total Parameters:    {params_m:.3f} M")
    print(f"FLOPs:       {flops_g:.3f} G")
    print(f"Average Inference Time: {avg_time:.4f} s")
    print("=" * 50)
