import copy
import math
import torch
from torch import nn
from .Encoder import Encoder
from .Decoder import *
from einops import reduce
class launch_seg(nn.Module):
    def __init__(self, in_ch=3, start_ch=48, num_classes=9, img_size=224, granular_sizes=[224,112,56], up_factor=[1,2,4], loops=[1,2,3,4]):
        super(launch_seg, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.granular_sizes = granular_sizes
        if granular_sizes is None and img_size == 224:
            self.granular_sizes = [img_size, img_size//2, img_size//4]
        elif granular_sizes is None and img_size == 256:
            self.granular_sizes = [img_size, img_size // 2, img_size // 4, img_size // 8]
        self.encoder = Encoder(in_ch=in_ch, start_ch=start_ch,
                               granular_sizes=self.granular_sizes, up_factor=up_factor, loops=loops)
        self.fc = nn.Linear(start_ch * 16, num_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _ = self.encoder(x)
        x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from thop import profile

    model = launch_seg(3, 48, 9, 224)
    dummy_input = torch.rand(1, 3, 224, 224)  # 1 batch
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.4fG, params: %.4fM' % (flops / 1000000000, params / 1000000))
