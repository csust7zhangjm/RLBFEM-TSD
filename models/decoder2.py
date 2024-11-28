# neck

import torch
import torch.nn as nn
from utils.conv import *


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels = 256,
                 mid_channels = 64,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.combine = nn.Sequential(
            ConvReLU(in_channels, mid_channels, 1, padding=0),
            ConvReLU(mid_channels, mid_channels, 3, padding=dilation, dilation=dilation),
            ConvReLU(mid_channels, in_channels, 1, padding=0))

    def forward(self, x):
        identity = x
        x = self.combine(x)
        out = x + identity
        return out

# dilated Encoder
# 串联的空间卷积
class DilatedEncoder(nn.Module):
    def __init__(self, in_channels, block_dilations=[1, 2, 3]):
        super().__init__()
        self.in_channel = in_channels
        self.block_dilation = block_dilations
        self.num_residual_blocks = len(self.block_dilation)
        self.bottleneck = nn.Sequential(
            Bottleneck(in_channels=self.in_channel, dilation=self.block_dilation[0]),
            Bottleneck(in_channels=self.in_channel, dilation=self.block_dilation[1]),
            Bottleneck(in_channels=self.in_channel, dilation=self.block_dilation[2])
        )

    def forward(self, x):
        # num_residual_blocks = len(self.block_dilation)
        # for i in range(num_residual_blocks):
        #     x = self.bottleneck(x)
        x = self.bottleneck(x)
        return x


# 解码器结构
class Decoder(nn.Module):
    def __init__(self, channels_list=[256, 256, 256, 256, 256]):
        super().__init__()
        self.block1 = DilatedEncoder(in_channels=channels_list[0])
        self.block2 = DilatedEncoder(in_channels=channels_list[1])
        self.block3 = DilatedEncoder(in_channels=channels_list[2])
        self.block4 = DilatedEncoder(in_channels=channels_list[3])
        self.block5 = DilatedEncoder(in_channels=channels_list[4])
        self.conv1 = ConvReLU(channels_list[0], channels_list[0], 3, stride=2, padding=1)  #80==>40
        self.conv2 = ConvReLU(channels_list[0], channels_list[0], 3, stride=2, padding=1)  #40==>20
        self.conv3 = ConvReLU(channels_list[0], channels_list[0], 3, stride=2, padding=1)  #20==>10
        self.conv4 = ConvReLU(channels_list[0], channels_list[0], 3, stride=2, padding=1)  #10==>5

    def forward(self, x):
        # x3, x4, x5, x6, x7 = x
        x3, x4, x5 = x
        t3 = self.block1(x3)
        t4 = self.block2(torch.add(self.conv1(t3), x4))
        t5 = self.block3(torch.add(self.conv2(t4), x5))
        
        # t6 = self.block4(torch.add(self.conv3(t5), x6))
        # t7 = self.block5(torch.add(self.conv4(t6), x7))
        # return t3, t4, t5, t6, t7
        return t3, t4, t5



# ASPP结构
# 并联的空洞卷积结构
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  #不用管没用！
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, in_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode ='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net






if __name__ == '__main__':
    x0 = torch.randn([1, 128, 64, 64])
    x1 = torch.randn([1, 256, 32, 32])
    x2 = torch.randn([1, 512, 16, 16])
    x = [x0, x1, x2]
    decoder = Decoder()
    out = decoder(x)
    for o in out:
        print(o.shape)

# torch.Size([1, 512, 16, 16])