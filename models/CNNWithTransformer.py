from core.CNN import *
from core.SwinTransformer import SwinTransformer
from core.Transformer import *
from core.AWB import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = MobileNetV2(num_classes=3)
        self.encoder2 = SwinTransformer([608, 608], depths=[2, 2, 6, 2])

        self.fuse1 = FCM(128)
        self.fuse2 = FCM(256)
        self.fuse3 = FCM(512)

        self.Conv = nn.Sequential(S_conv_r(1024, 512),
                                  nn.BatchNorm2d(512),
                                  nn.GELU())
        # self.decoder = Decoder()
        # self.enhance = SASPP(512, ratio=[1, 3, 5, 7])

        #通道注意力
        # self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x1, x2, x3 = self.encoder1(x)
        y1, y2, y3 = self.encoder2(x)

        f1 = self.fuse1(x1, y1)
        f2 = self.fuse2(x2, y2)
        f3 = self.fuse3(x3, y3)

        # #通道注意力
        # B1, C1, H1, W1 = out1.shape
        # B2, C2, H2, W2 = out2.shape
        # x_temp = self.avg(out1)
        # y_temp = self.avg(out2)
        # x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        # y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        # x_temp = out1.permute(0, 2, 3, 1)
        # y_temp = out2.permute(0, 2, 3, 1)
        # x1 = x_temp * x_weight
        # y1 = y_temp * y_weight
        # # out = x1 + y1
        # # out = out.permute(0, 3, 1, 2)
        # x1 = x1.permute(0, 3, 1, 2)
        # y1 = y1.permute(0, 3, 1, 2)
        #
        # # out = out1 + out2
        # out = torch.cat([x1, y1], dim=1)
        # out = self.Conv(out) #torch.Size([1, 256, 16, 16])
        # out = self.enhance(out)

        return f1, f2, f3

if __name__ == '__main__':
    x = torch.rand(1, 3, 608, 608)
    net = Net()
    out = net(x)
    for o in out:
        print(o.shape)

# torch.Size([1, 128, 80, 80])
# torch.Size([1, 256, 40, 40])
# torch.Size([1, 512, 20, 20])