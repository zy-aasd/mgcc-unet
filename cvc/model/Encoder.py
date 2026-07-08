import torch
import torch.nn as nn
import torch.nn.functional as F
from .GranularBall import GranularBallLayer
from .GranularInteraction import GranularBallInteraction

def channel_shuffle(x, groups: int):
    B, C, H, W = x.size()
    channels_per_group = C // groups
    # reshape
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(B, -1, H, W)
    return x

class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1d, self).__init__()
        self.conv = nn.Conv1d(
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
        # 生成空间注意力权重
        attention = self.conv(combined)  # batch, 1, H, W
        attention = self.sigmoid(attention)
        return x * attention

class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels, use_dw=False):
        super(ConvStem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels) if use_dw
            else nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, start_ch=48, granular_sizes=[224, 112, 56], up_factor=[1,2,4], loops=[1,2,3,4]):
        super(Encoder, self).__init__()
        self.loops = loops

        self.conv1 = ConvStem(in_ch, start_ch) # 24
        self.granular_ball1 = GranularBallLayer(start_ch, granular_sizes, groups=2, up_factor=up_factor)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = ConvStem(start_ch, start_ch * 2) # 48
        self.granular_ball2 = GranularBallLayer(start_ch * 2, [x // 2 for x in granular_sizes], groups=4, up_factor=up_factor)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = ConvStem(start_ch * 2, start_ch * 4) # 96
        self.granular_ball3 = GranularBallLayer(start_ch * 4, [x // 4 for x in granular_sizes], groups=8, up_factor=up_factor)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4 = ConvStem(start_ch * 4, start_ch * 8) # 192
        self.granular_ball4 = GranularBallLayer(start_ch * 8, [x // 8 for x in granular_sizes], groups=12, up_factor=up_factor)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5 = ConvStem(start_ch * 8, start_ch * 16) # 384

        self.inter4 = GranularBallInteraction(start_ch * 8, loop=loops[-1])
        self.inter3 = GranularBallInteraction(start_ch * 4, loop=loops[-2])
        self.inter2 = GranularBallInteraction(start_ch * 2, loop=loops[-3])
        self.inter1 = GranularBallInteraction(start_ch * 1, loop=loops[-4])

    def forward(self, x):
        x_skip = []
        x = self.conv1(x) # B C 224 224
        x_skip.append(x) # 0
        x_gb = self.granular_ball1(x) # B C 224 224
        if self.loops[0] != 0:
            x = self.inter1(x_gb) + x # B C 224 224
        x = self.pool1(x) # B C 112 112

        x = self.conv2(x) # B 2C 112 112
        x_skip.append(x) # 1
        x_gb = self.granular_ball2(x) # B 2C 112 112
        if self.loops[1] != 0:
            x = self.inter2(x_gb) + x # B 2C 112 112
        x = self.pool2(x) # B 2C 56 56

        x = self.conv3(x) # B 4C 56 56
        x_skip.append(x) # 2
        x_gb = self.granular_ball3(x) # B 4C 56 56
        if self.loops[2] != 0:
            x = self.inter3(x_gb) + x # B 4C 56 56
        x = self.pool3(x) # B 4C 28 28

        x = self.conv4(x)  # B 8C 28 28
        x_skip.append(x) # 3
        x_gb = self.granular_ball4(x)  # B 8C 28 28
        if self.loops[3] != 0:
            x = self.inter4(x_gb) + x  # B 8C 28 28
        x = self.pool4(x) # B 8C 14 14

        x = self.conv5(x) # B 16C 14 14
        return x, x_skip

if __name__ == "__main__":
    from thop import profile
    model = Encoder(3, 24)
    dummy_input = torch.rand(1, 3, 224, 224)  # 1 batch
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.4fG, params: %.4fM' % (flops / 1000000000, params / 1000000))
