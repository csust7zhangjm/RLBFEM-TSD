# Attention Weighting Block  注意力加权结构

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conv import ConvReLU


class AttentionWeightingBlock(nn.Module):
    def __init__(self, inchannel):
        super(AttentionWeightingBlock, self).__init__()
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        # self.gmp = nn. AdaptiveMaxPool2d((1, 1)) # 全局最大池化
        self.conv11 = nn.Conv2d(inchannel, inchannel, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(inchannel, inchannel // 4, 1, groups=inchannel // 4)

    def forward(self, x):
        x1 = self.sigmoid(self.conv11(F.adaptive_avg_pool2d(x, output_size=(1, 1))))   # [1, 64, 1, 1]
        x2 = self.sigmoid(self.conv11(F.adaptive_max_pool2d(x, output_size=(1, 1))))   # [1, 64, 1, 1]
        x_features = x1 + x2   # [1, 64, 1, 1]
        y_features = torch.mul(x_features, x)   # [1, 64, 608, 608]
        y_residual = x + y_features   # [1, 64, 608, 608]
        y = self.conv(y_residual)   # [1, 32, 608, 608]
        return y


class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__() #没用拉
        # self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, dim)
        self.down = ConvReLU(3 * dim, dim, kernel_size=1)

        self.fuse = nn.Sequential(ConvReLU(3 * dim, dim, 1),  # 1x1Conv
                                  ConvReLU(dim, dim, 3, groups=dim, padding=1),  # 3x3DWconv
                                  ConvReLU(dim, dim, 3, groups=dim, padding=1),  # 3x3DWconv
                                  ConvReLU(dim, dim, 3, groups=dim, padding=1))  # 3x3DWconv

    def forward(self, x1, y1):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = y1.shape
        # 两部分输入首先进行通道注意力，抑制影响力较低的通道
        x_temp = self.avg(x1)
        y_temp = self.avg(y1)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = x1.permute(0, 2, 3, 1)
        y_temp = y1.permute(0, 2, 3, 1)

        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        out1 = torch.cat([x1, y1], dim=3)

        out2 = x1 * y1

        fuse = torch.cat([out1, out2], dim=3)
        fuse = fuse.permute(0, 3, 1, 2)

        out = self.fuse(fuse)
        out = out + self.down(fuse)

        return out


if __name__ == '__main__':
    x = torch.randn([1, 64, 608, 608])
#     ATB = AttentionWeightingBlock(64)
    ATB = FCM(64)
    out = ATB(x, x)
    print(out.shape)


