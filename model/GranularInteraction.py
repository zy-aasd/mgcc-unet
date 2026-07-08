import torch.nn as nn
import torch.nn.functional as F
import torch

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        # Add both outputs
        out = avg_out + max_out
        # Apply sigmoid
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
        # Multiply with the input
        return x * out + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # b, 1, H, W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # b, 1, H, W
        combined = torch.cat([avg_out, max_out], dim=1)  # b, 2, H, W
        attention = self.conv(combined)  # batch, 1, H, W
        attention = self.sigmoid(attention)
        return x * attention

class GranularBallInteraction(nn.Module):
    def __init__(self, in_channels, loop=1, show=False):
        super(GranularBallInteraction, self).__init__()
        self.ca = ChannelAttention(in_channels//4)
        self.sa = SpatialAttention()
        self.convs_expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
        )
        self.convs_squeeze = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels, kernel_size=1, stride=1, padding=0),
        )
        self.loop = loop
        self.show = show
    def forward(self, x):
        for i in range(self.loop):
            x = self.convs_expand(x)
            x, y = torch.chunk(x, 2, dim=1)
            x = self.ca(x)
            y = self.sa(y)
            x = torch.concat([x, y], dim=1)
            x = self.convs_squeeze(x)
        return x
