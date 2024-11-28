# This file contains modules common to various models
import math
from typing import Sized
import torch
from torch import nn
from torch.nn import functional as F

from utils.diversebranchblock import DiverseBranchBlock
# import diversebranchblock.DiverseBranchBlock
from utils.utilss import *

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Conv, self).__init__()
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # padding
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class new_Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(new_Conv, self).__init__()
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # padding
        self.conv = DiverseBranchBlock(c1, c2, k, s, p, groups=g)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act((self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class new_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(new_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = new_Conv(c1, c_, 1, 1)
        self.cv2 = new_Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CA_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(CA_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = new_Conv(c1, c_, 1, 1)
        self.cv2 = new_Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # if(self.add==True):
        #     x1 = x2 = c1
        # else:
        #     x1 = x2 = c2
        self.ca = CoordAtt(c2,c2,16)

    def forward(self, x):
        #return self.ca(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        #return self.ca(x) + self.cv2(self.cv1(x)) if self.add else self.ca(self.cv2(self.cv1(x)))
        return self.ca(self.cv2(self.cv1(x))) #不使用add ->0.823

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class new_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(new_BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = new_Conv(c1, c_, 1, 1)
        #self.cv2 = new_Conv(c1, c_, 1, 1)
        #self.cv3 = new_Conv(c_, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = new_Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[new_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        #return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        #return self.cv4(self.act(torch.cat((y1, y2), dim=1)))
        return self.cv4(torch.cat((y1, y2), dim=1))

class CA_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CA_BottleneckCSP, self).__init__()
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

# No cv4 and source conv
class WeightBottleneckCSP(nn.Module):
    # Abalation
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(WeightBottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.act(self.bn(torch.cat((y1, y2), dim=1)))
        #return self.act(torch.cat((y1, y2), dim=1))

#no cv4 and dbb and CA concat
class Weight_CA_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Weight_CA_BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = new_Conv(c1, c_, 1, 1)
        self.ca = CoordAtt(c_,c_,16)
        #self.cv2 = new_Conv(c1, c_, 1, 1)
        #self.cv3 = new_Conv(c_, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = new_Conv(c2, c2, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.m = nn.Sequential(*[CA_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # source is CA_Bottleneck
        
    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        #y2 = self.cv2(self.ca(x))
        y2 = self.cv2(x)

        #return self.act(self.bn(torch.cat((y1, y2), dim=1)))
        #return self.act(torch.cat((y1, y2), dim=1))
        return torch.cat((y1, y2), dim=1) #source

# 用这个module 代替bottleneckCSP中的1*1卷积
class PreWeighting(nn.Module):
    def __init__(self,in_channels,out_channels,ratio=16):
        super(PreWeighting,self).__init__()
        self.channels = in_channels
        self.conv1 = DiverseBranchBlock(
            in_channels=in_channels,
            out_channels=int(in_channels / ratio),
            kernel_size=1,
            stride=1)
        self.conv2 = DiverseBranchBlock(
            in_channels=int(in_channels / ratio),
            out_channels=out_channels,
            kernel_size=1,
            stride=1)
        self.bn1 = nn.BatchNorm2d(int(in_channels / ratio))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.act2 = nn.Sigmoid()
    def forward(self, x):
    	# mini_size即为当前stage中最小分辨率的shape：H_s, W_s
        mini_h = mini_w = 20  # H_s, W_s
        # 将所有stage的input均压缩至最小分辨率，由于最小的一个stage的分辨率已经是最小的了
        # 因此不需要进行压缩
        # out需要从stage取，x是原featuremap
        out = F.adaptive_avg_pool2d(x, (mini_h,mini_w))
        out = self.act1(self.conv1(out))
        out = self.act2(self.conv2(out))  # sigmoid激活
        return out

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        return x[-1] + x[-2]

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        #print(x[-1].shape)
        #print(x[-2].shape)
        return x[-1] * x[-2]

# from -1 -> upsameple 
class Adaptiveupsample(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self):
        super(Adaptiveupsample, self).__init__()
    def forward(self, x):
        needh = x[-1].shape[2]
        needw = x[-1].shape[3]
        x = F.interpolate(x[-2], size=[needh,needw], mode='nearest')
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# 重载CA or 不重载
class CoordAtt(nn.Module):#用了
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        #self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv1 = DiverseBranchBlock(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x  #[H,W,C]
        
        n,c,h,w = x.size()
        x_h = F.adaptive_avg_pool2d(x, output_size=(None, 1))#self.pool_h(x)
        # print("x_h:", x_h.shape)  #[H,1,C]
        x_w = F.adaptive_avg_pool2d(x, output_size=(1, None)).permute(0, 1, 3, 2) #self.pool_w(x).permute(0, 1, 3, 2)
        # print("x_w:", x_w.shape)  #[1,W,C]  ==> [H,1,C]

        y = torch.cat([x_h, x_w], dim=2)  # [2H,1,C]
        y = self.conv1(y)
        #y = self.bn1(y)
        y = self.act(y)   # [2H,1,C']
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [H,1,C'] ==> [1,H,C'] ==> [1,W,C']

        a_h = self.conv_h(x_h).sigmoid()  # [H,1,C']
        a_w = self.conv_w(x_w).sigmoid()  # [1,W,C']

        out = identity * a_w * a_h     # [H,W,C]

        return out

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3): #没用！
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # 最后两维进行转置删除维度一，再转置，再添加一维

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




if __name__ == '__main__':
    co = CoordAtt(32, 32)
    x = torch.randn([1, 32, 608, 608])
    out = co(x)
    print(out.shape)
