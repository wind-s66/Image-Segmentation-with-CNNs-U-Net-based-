
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([double_conv(1 if i == 0 else widths[i-1], w)
                                      for i, w in enumerate(widths)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i-1], 2, 2)
                                 for i in range(4, 0, -1)])
        self.decoders = nn.ModuleList([double_conv(widths[i-1]*2, widths[i-1])
                                      for i in range(4, 0, -1)])
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)
