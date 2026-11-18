"""Attention Residual U-Net adapted from DeepPipeline's encoder/decoder design.
Original reference: https://github.com/lext/deep-pipeline
New features: residual convolution blocks, attention-gated skips, spatial bottleneck attention.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(cout))
        self.shortcut = (nn.Identity() if cin == cout else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels):
        super().__init__()
        mid = max(1, skip_channels // 2)
        self.skip_proj = nn.Conv2d(skip_channels, mid, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, mid, 1, bias=False)
        self.score = nn.Conv2d(mid, 1, 1)

    def forward(self, skip, gate):
        gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        weights = torch.sigmoid(self.score(F.relu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class UNet(nn.Module):
    def __init__(self, base_channels=16, depth=5, dropout=0.2):
        super().__init__()
        if depth < 2 or base_channels < 1:
            raise ValueError('depth must be >= 2 and base_channels >= 1')
        widths = [base_channels * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList([
            ResidualBlock(1 if i == 0 else widths[i-1], w)
            for i, w in enumerate(widths)
        ])
        self.center = ResidualBlock(widths[-1], widths[-1])
        self.spatial_attention = SpatialAttention()
        self.dropout = nn.Dropout2d(dropout)
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = widths[-1]
        for w in reversed(widths):
            self.gates.append(AttentionGate(w, current))
            self.decoders.append(ResidualBlock(current + w, w))
            current = w
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            skip = encoder(x)
            skips.append(skip)
            x = F.max_pool2d(skip, 2)
        x = self.dropout(self.spatial_attention(self.center(x)))
        for gate, decoder, skip in zip(self.gates, self.decoders, reversed(skips)):
            filtered_skip = gate(skip, x)
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, filtered_skip], dim=1))
        return self.head(x)
