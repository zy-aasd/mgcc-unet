import torch.nn as nn
import torch.nn.functional as F
import torch

def channel_shuffle(x, groups: int):
    B, C, H, W = x.size()
    channels_per_group = C // groups
    # reshape
    x = x.view(B, groups, channels_per_group, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(B, -1, H, W)
    return x


# granular_sizes=[32, 16, 8] sorted by [img_size,img_size//2,img_size//4]
class GranularBallLayer(nn.Module):
    def __init__(self, in_channels, granular_sizes=[224, 112, 56], groups=2, up_factor=[1,2,4]):
        super().__init__()
        self.granular_sizes = granular_sizes
        self.len_gb = len(granular_sizes)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_channels, in_channels//12, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels//12, in_channels, 1, bias=False)
        self.up_factor = up_factor

        self.adaptive_maxpools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((size, size)) for size in granular_sizes
        ])
        self.adaptive_avgpools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((size, size)) for size in granular_sizes
        ])
        # self.conv_trans = nn.ModuleList([
        #     nn.Upsample(scale_factor=up_factor[i], mode='bicubic')
        #     for i in range(len(granular_sizes))
        # ])
        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose2d(in_channels // len(granular_sizes), in_channels // len(granular_sizes), up_factor[i],
                               up_factor[i], 0) for i in range(len(granular_sizes))
        ])

        self.extract = nn.Conv2d(in_channels, in_channels // len(granular_sizes), 1,1,0)
        self.mixture = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.groups = groups
    def forward(self, x):
        # input:224x224  gb:224,112,56   224->Global 112 56->Local
        granular_features = []
        for i, size in enumerate(self.granular_sizes):
            # 3 // 2 = 1
            if size == x.shape[2]:
                pooled = self.fc2(self.act(self.fc1(self.adaptive_avgpools[i](x))))
            else:
                pooled = self.fc2(self.act(self.fc1(self.adaptive_maxpools[i](x))))
            pooled = self.extract(pooled)
            upsampled = pooled
            upsampled = self.conv_trans[i](upsampled)
            # nearest bilinear bicubic
            # upsampled = F.interpolate(upsampled, scale_factor=self.up_factor[i], mode='nearest')
            # upsampled = F.interpolate(upsampled, scale_factor=self.up_factor[i], mode='bilinear', align_corners=True)
            # upsampled = F.interpolate(upsampled, scale_factor=self.up_factor[i], mode='bicubic', align_corners=True)
            # 记录该粒球特征
            granular_features.append(upsampled)
        # 多粒度特征融合
        mix = torch.cat(granular_features, dim=1)
        mix = self.mixture(mix)
        return mix

