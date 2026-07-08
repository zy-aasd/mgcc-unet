import torch
import torch.nn as nn
from .GranularBall import GranularBallLayer
from .GranularInteraction import GranularBallInteraction
import torch.nn.functional as F

def channel_shuffle(x, groups: int):
    B, C, H, W = x.size()
    channels_per_group = C // groups
    # reshape
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(B, -1, H, W)
    return x

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

class MSUP(nn.Module):
    def __init__(self, dim, group=2):
        super(MSUP, self).__init__()
        self.g1_conv11 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
        self.group = group
        self.dw33 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)
        self.dw55 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=1)
        self.dw11 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1)
        self.g2_conv11 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=1),
            nn.BatchNorm2d(dim),
        )
        self.g3_conv11 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x, x_skip):
        x_org = x
        x = self.g1_conv11(x)
        x = channel_shuffle(x, self.group)
        org = x
        x = self.dw33(org)
        z = self.dw55(org)
        y = self.dw11(org)
        x = org + x + z + y
        x = self.g2_conv11(x)
        x_pool = self.avg(x_org)
        x = torch.cat((x_pool, x), dim=1)
        x = self.g3_conv11(x) + x_org
        x = self.up(x) + x_skip
        return x


class Decoder(nn.Module):
    def __init__(self, out_ch=9, start_ch=24):
        super(Decoder, self).__init__()
        self.conv4 = ConvStem(start_ch * 16, start_ch * 8) # 192
        self.up4 = MSUP(start_ch * 8, group=16)

        self.conv3 = ConvStem(start_ch * 8, start_ch * 4) # 96
        self.up3 = MSUP(start_ch * 4, group=8)

        self.conv2 = ConvStem(start_ch * 4, start_ch * 2) # 48
        self.up2 = MSUP(start_ch * 2, group=4)

        self.conv1 = ConvStem(start_ch * 2, start_ch) # 24
        self.up1 = MSUP(start_ch, group=2)

        self.seghead = nn.Conv2d(start_ch, out_ch, 1)

    def forward(self, x, x_skip):
        x = self.conv4(x) # B 8C 14 14
        x = self.up4(x, x_skip[3]) # B 8C 28 28

        x = self.conv3(x) # B 4C 28 28
        x = self.up3(x, x_skip[2]) # B 4C 56 56

        x = self.conv2(x) # B 2C 56 56
        x = self.up2(x, x_skip[1]) # B 2C 112 112

        x = self.conv1(x) # B C 112 112
        x = self.up1(x, x_skip[0]) # B C 224 224

        x = self.seghead(x)
        return x

class DecoderGB(nn.Module):
    def __init__(self, out_ch=9, start_ch=24, granular_sizes=[224, 112, 56], up_factor=[1,2,4], loops=[1,2,3,4]):
        super(DecoderGB, self).__init__()
        self.loops = loops
        self.stem4 = ConvStem(start_ch * 16, start_ch * 8) # 192
        self.granular_ball4 = GranularBallLayer(start_ch * 8, [x // 8 for x in granular_sizes], groups=12, up_factor=up_factor)
        self.up4 = nn.ConvTranspose2d(start_ch * 8, start_ch * 8, 2, stride=2)

        self.stem3 = ConvStem(start_ch * 8, start_ch * 4) # 96
        self.granular_ball3 = GranularBallLayer(start_ch * 4, [x // 4 for x in granular_sizes], groups=8, up_factor=up_factor)
        self.up3 = nn.ConvTranspose2d(start_ch * 4, start_ch * 4, 2, stride=2)

        self.stem2 = ConvStem(start_ch * 4, start_ch * 2) # 48
        self.granular_ball2 = GranularBallLayer(start_ch * 2, [x // 2 for x in granular_sizes], groups=4, up_factor=up_factor)
        self.up2 = nn.ConvTranspose2d(start_ch * 2, start_ch * 2, 2, stride=2)

        self.stem1 = ConvStem(start_ch * 2, start_ch) # 24
        self.granular_ball1 = GranularBallLayer(start_ch, granular_sizes, groups=2, up_factor=up_factor)
        self.up1 = nn.ConvTranspose2d(start_ch, start_ch, 2, stride=2)

        self.seghead = nn.Conv2d(start_ch, out_ch, 1)

        self.inter4 = GranularBallInteraction(start_ch * 8, loop=loops[-1])
        self.inter3 = GranularBallInteraction(start_ch * 4, loop=loops[-2])
        self.inter2 = GranularBallInteraction(start_ch * 2, loop=loops[-3])
        self.inter1 = GranularBallInteraction(start_ch * 1, loop=loops[-4])

    def forward(self, x, x_skip):
        x = self.stem4(x)
        x = self.up4(x) + x_skip[3]
        x_gb = self.granular_ball4(x)
        if self.loops[-1] != 0:
            x = self.inter4(x_gb) + x  # B 2C 112 112

        x = self.stem3(x)
        x = self.up3(x) + x_skip[2]# 56
        x_gb = self.granular_ball3(x)
        if self.loops[-2] != 0:
            x = self.inter3(x_gb) + x


        x = self.stem2(x)
        x = self.up2(x) + x_skip[1]# 28
        x = x + x_skip[-3]
        x_gb = self.granular_ball2(x)
        if self.loops[-3] != 0:
            x = self.inter2(x_gb) + x


        x = self.stem1(x)
        x = self.up1(x) + x_skip[0]# 28
        x = x + x_skip[-4]
        x_gb = self.granular_ball1(x)
        if self.loops[-4] != 0:
            x = self.inter1(x_gb) + x

        x = self.seghead(x)
        return x


if __name__ == "__main__":
    from thop import profile
    model = Decoder(9, 24)
    x_skip = []
    x_skip.append(torch.randn(1, 24, 112, 112))
    x_skip.append(torch.randn(1, 48, 56, 56))
    x_skip.append(torch.randn(1, 96, 28, 28))
    x_skip.append(torch.randn(1, 192, 14, 14))

    dummy_input = torch.rand(1, 384, 14, 14)  # 1 batch
    flops, params = profile(model, (dummy_input, x_skip))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.4fG, params: %.4fM' % (flops / 1000000000, params / 1000000))


