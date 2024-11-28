import torch
from torch import nn

from core.SwinTransformer import SwinTransformerBlock
from utils.conv import *
# from core.AWB import *
from utils.common import *
# from utils.dbb_transforms import *
from utils.diversebranchblock import *
from models.mamba import VSSBlock

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvSiLU(c1, c_, 1, 1)
        self.cv2 = ConvSiLU(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class C3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(C3, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = ConvSiLU(c1, c_, 1, 1)
#         self.cv2 = ConvSiLU(c1, c_, 1, 1)
#         self.cv3 = ConvSiLU(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat(
#             (
#                 self.m(self.cv1(x)),
#                 self.cv2(x)
#             )
#             , dim=1))


class C3(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = new_Conv(c1, c_, 1, 1)
        self.ca = CoordAtt(c1,c1,16)
        #self.cv2 = new_Conv(c1, c_, 1, 1)
        #self.cv3 = new_Conv(c_, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = new_Conv(c2, c2, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.m = nn.Sequential(*[new_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(self.ca(x))

        #return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        #return self.cv4(self.act(torch.cat((y1, y2), dim=1)))
        return self.cv4(torch.cat((y1, y2), dim=1))


class Neck(nn.Module):
    def __init__(self, in_chans=[128, 256, 512]):
        super().__init__()
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = ConvSiLU(in_chans[2], in_chans[1], 1, 1)
        self.conv3_for_upsample1 = C3(in_chans[2], in_chans[1], 3, shortcut=False)
        # self.conv3_for_upsample1    = SwinTransformerBlock(512, input_resolution=[38, 38], num_heads=16)
        # self.conv55 = ConvSiLU(in_chans[2], in_chans[1], 1)

        self.conv_for_feat2         = ConvSiLU(in_chans[1], in_chans[0], 1, 1)
        self.conv3_for_upsample2    = SwinTransformerBlock(256, input_resolution=[76, 76], num_heads=16)
        self.conv44 = ConvSiLU(in_chans[1], in_chans[0], 1)

        self.down_sample1           = ConvSiLU(in_chans[0], in_chans[0], 3, 2)
        self.conv3_for_downsample1  = SwinTransformerBlock(256, input_resolution=[38, 38], num_heads=16)

        self.down_sample2           = ConvSiLU(in_chans[1], in_chans[1], 3, 2)
        # self.conv3_for_downsample2  = SwinTransformerBlock(512, input_resolution=[19, 19], num_heads=16)
        self.conv3_for_downsample2 = C3(in_chans[2], in_chans[2], 3, shortcut=False)

        
        # self.mamba3 = VSSBlock(hidden_dim=in_chans[0])
        self.mamba4 = VSSBlock(hidden_dim=in_chans[1])
        self.mamba5 = VSSBlock(hidden_dim=in_chans[2])
        

    def forward(self, x):
        c3, c4, c5 = x
        
        # x_size = (c3.shape[2], c3.shape[3])
        # c3 = self.mamba3(c3, x_size)
        
        x_size = (c4.shape[2], c4.shape[3])
        c4 = self.mamba4(c4, x_size)
        
        x_size = (c5.shape[2], c5.shape[3])
        c5 = self.mamba5(c5, x_size)

        # 16, 16, 512 ==> 16, 16, 256
        P5 = self.conv_for_feat3(c5)
        # 16, 16, 256 -> 32, 32, 256
        P5_upsample = self.upsample(P5)
        # 32, 32, 512
        P4 = torch.cat([P5_upsample, c4], 1)
        # 32, 32, 256
        # B, C, H, W = P4.shape
        # P4 = P4.view([B, C, -1]).permute(0, 2, 1).contiguous()
        # P4 = self.conv3_for_upsample1(P4)
        # P4 = P4.view([B, H, W, C]).permute(0, 3, 1, 2).contiguous()
        # P4 = self.conv55(P4)
        P4 = self.conv3_for_upsample1(P4)

        # 32, 32, 256 -> 32, 32, 128
        P4 = self.conv_for_feat2(P4)
        # 64, 64, 128
        P4_upsample = self.upsample(P4)
        # 64, 64, 256
        P3 = torch.cat([P4_upsample, c3], 1)
        # 64, 64, 128
        B, C, H, W = P3.shape

        # print("P3 start!!!")
        # print(B, C, H, W)
        # print("P3 End!!!")

        P3 = P3.view([B, C, -1]).permute(0, 2, 1).contiguous()
        P3 = self.conv3_for_upsample2(P3)
        P3 = P3.view([B, H, W, C]).permute(0, 3, 1, 2).contiguous()
        
        # P3 = self.mamba1(P3, (H, W))



        P3 = self.conv44(P3)

        # 64, 64, 128 ==> 32, 32, 128
        P3_downsample = self.down_sample1(P3)
        # 32, 32, 256
        P4 = torch.cat([P3_downsample, P4], 1)
        # 32, 32, 256
        B, C, H, W = P4.shape

        # print("P4 start!!!")
        # print(B, C, H, W)
        # print("P4 End!!!")

        P4 = P4.view([B, C, -1]).permute(0, 2, 1).contiguous()
        P4 = self.conv3_for_downsample1(P4)
        P4 = P4.view([B, H, W, C]).permute(0, 3, 1, 2).contiguous()

        # P4 = self.mamba2(P4, (H, W))

        # 32, 32, 256 ==> 16, 16, 256
        P4_downsample = self.down_sample2(P4)
        # 16, 16, 512
        P5 = torch.cat([P4_downsample, P5], 1)
        # 16, 16, 512
        # B, C, H, W = P5.shape
        # P5 = P5.view([B, C, -1]).permute(0, 2, 1).contiguous()
        # P5 = self.conv3_for_downsample2(P5)
        # P5 = P5.view([B, H, W, C]).permute(0, 3, 1, 2).contiguous()
        P5 = self.conv3_for_downsample2(P5)

        return P3, P4, P5


if __name__ == '__main__':
    x1 = torch.randn([1, 128, 76, 76])
    x2 = torch.randn([1, 256, 38, 38])
    x3 = torch.randn([1, 512, 19, 19])
    net = Neck()
    out = net([x1, x2, x3])
    for o in out:
        print(o.shape)