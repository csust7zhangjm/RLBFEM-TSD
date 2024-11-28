import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F

from utils.conv import *
BasicBlock = ConvReLU

def conv_flops(k, c_in, c_out, stride, padding, resolution, bias=True, dialation=1, groups=1):
    batch_size = 1
    output_dims = math.floor((resolution + 2 * padding - dialation * (k - 1) - 1) / stride + 1)
    kernel_dims = k
    in_channels = c_in
    out_channels = c_out
    # groups = 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(kernel_dims ** 2) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(output_dims ** 2)

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class RelPosAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.input_resolution[0])
        coords_w = torch.arange(self.input_resolution[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.input_resolution[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.input_resolution[1] - 1
        relative_coords[:, :, 0] *= 2 * self.input_resolution[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.input_resolution[0] * self.input_resolution[1], self.input_resolution[0] * self.input_resolution[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class HiLo(nn.Module):
    """
    HiLo Attention

    Link: https://arxiv.org/abs/2205.13213
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2,
                 alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape
        # if old_H % self.ws == 0 or old_W % self.ws != 0:
        #     x = F.pad(x, (0, self.ws - old_W % self.ws, 0, self.ws - old_H % self.ws), "constant", 0)
        #     B, H, W, C = x.shape
        # else:
        #     H, W = old_H, old_W
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1,
                                                                                                              4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        # 重新将x放缩回原来的大小
        # x = F.adaptive_avg_pool2d(x, (old_H, old_W))
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)
        return x

    def flops(self, N):
        H = int(N ** 0.5)
        # when the height and width cannot be divided by ws, we pad the feature map in the same way as Swin Transformer for object detection/segmentation
        Hp = Wp = self.ws * math.ceil(H / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.h_dim * 3
        nW = Np / self.ws / self.ws
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.h_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.h_dim * self.h_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.l_dim
        # H = int(Np ** 0.5)
        kv_len = (Hp // self.ws) ** 2
        # k, v
        lofi_flops += kv_len * self.dim * self.l_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.l_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.l_dim * self.l_dim

        return hifi_flops + lofi_flops

class PatchEmbed(nn.Module):
    def __init__(self, dim, p_size):
        super().__init__()
        self.embed = nn.Sequential(*[BasicBlock(in_channels=3, out_channels=dim, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, groups=dim),])
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.norm(self.embed(x)) #torch.Size([1, 64, 128, 128])
        return x

class PatchMerge(nn.Module):
    def __init__(self, inc, outc, stride = 2):
        super().__init__()
        self.merge = nn.Conv2d(inc, outc, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(outc)

    def forward(self, x):
        return self.norm(self.merge(x))

class Mlp(nn.Module):
    def __init__(self, inc, outc = None, dropout = 0.2):
        super().__init__()
        # outc = outc or inc * 2
        outc = outc or inc
        self.fc1 = nn.Conv2d(inc, outc, 1)
        self.fc2 = nn.Conv2d(outc, outc, 3, stride=1, padding=1, groups=outc)
        self.fc3 = nn.Conv2d(outc, inc, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x

class Block(nn.Module):
    def __init__(self, inc, window_size=8, num_head=8, alpha = 0.5, dropout = 0.):
        super().__init__()
        self.norm = nn.BatchNorm2d(inc)

        self.HiLo = HiLo(inc, num_heads=num_head, window_size=window_size, alpha=alpha)
        self.mlp = Mlp(inc, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x.reshape(B, C, -1).permute(0, 2, 1)
        x1 = self.HiLo(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + self.norm(x1)
        x = x + self.norm(self.mlp(x))

        return x


class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    #print(x.shape)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads=2, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #self.norm1 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class Transformer(nn.Module):
    def __init__(self, repeat_num=[2, 2, 6, 2]):
        super().__init__()
        #layer1
        self.embed = PatchEmbed(64, 4)
        # self.block1 = Block(64, window_size=2, num_head=8, alpha = 0.4, dropout=0.)
        self.block1 = nn.Sequential(*[SwinTransformerBlock(64, num_heads=4)] * repeat_num[0])
        #layer2
        self.merge2 = PatchMerge(64, 128)
        # self.block2 = Block(128, window_size=2, num_head=8, alpha=0.3, dropout=0.)
        self.block2 = nn.Sequential(*[SwinTransformerBlock(128, num_heads=4)] * repeat_num[1])
        #layer3
        self.merge3 = PatchMerge(128, 256)
        # self.block3 = Block(256, window_size=2, num_head=8, alpha=0.2, dropout=0.)
        self.block3 = nn.Sequential(*[SwinTransformerBlock(256, num_heads=8, shift_size=2),
                                      SwinTransformerBlock(256, num_heads=8)] * (repeat_num[2] // 2))

        #layer4
        self.merge4 = PatchMerge(256, 512)
        # self.block4 = Block(512, window_size=2, num_head=8, alpha=0.1, dropout=0.)
        self.block4 = self.block4 = nn.Sequential(*[SwinTransformerBlock(512, num_heads=8, shift_size=2),
                                                    SwinTransformerBlock(512, num_heads=8)])


    def forward(self, x):
        #layer1
        x1 = self.embed(x)
        x1 = self.block1(x1)  # torch.Size([1, 64, 128, 128])

        #layer2
        x2 = self.merge2(x1)
        x2 = self.block2(x2)  # torch.Size([1, 128, 64, 64])

        #layer3
        x3 = self.merge3(x2)
        x3 = self.block3(x3) #torch.Size([1, 256, 32, 32])

        #layer4
        x4 = self.merge4(x3) #torch.Size([1, 512, 16, 16])
        x4 = self.block4(x4) #torch.Size([1, 512, 16, 16])

        return x2, x3, x4

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    model = Transformer()
    x1, x2, x3= model(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    # torch.save(model.state_dict(), "swin.pth")

    # x = torch.Tensor(1, 16, 400, 600)
    # swin = SwinTransformerBlock(dim=16, num_heads=4)
    # x = swin(x)
    # print(x.shape)
