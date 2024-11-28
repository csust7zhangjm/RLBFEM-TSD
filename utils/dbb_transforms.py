import torch
import numpy as np
import torch.nn.functional as F

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std
    # bias = βj - μj*gammaj / σj
    # F = Fj * gammaj/sigmaj  (sigma = std)
def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      # input,weight
        # k2 -> E*D*3*3
        # k1 -> C*D*1*1 (D(output) is arbitrary,s C种通道为D的1*1卷积核),原本是D*C*1*1(C is input,D is output)的卷积运算
        # 由于Transfer 1是 1*1卷积只能通过通道线性组合(channel-wise linear combination but no spatial aggregation)
        # 线性重组k*K卷积核的参数然后合并到 k*k conv中
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        # b_hat = ΣDKK bd * F^2
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
        # 将结果分为g组然后执行转换2与转换3
    return k, b_hat + b2

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    # np.tile() 这里返回（0->inputdim-1)*groups的一维数组
    # F'd,c,:,: = 1/ k^2
    # d is ouput channel , c is input channel
    # same kernel size K and stride s applied to C channels
    # 返回的矩阵是对角线为 1.0 / 9 的 3*3矩阵，其余为 0(保证了当前输入的通道权重为1其他为0)
    return k

#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    # kernel.size(2)(3) 对应的kernel后两位 即 kernel H and W
    # 如 1*1 -> 3*3 (3-1)//2 = 1 即 pad = 1
    # 如 3*1 -> 3*3 pad h = 0 ,pad w = 1
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])
    # 通过0填充将一个kh*kw -> K*K
    # 如 1*1 卷积 通过对kernel size和feature map 进行 0 填充可以等价一个 3*3卷积
    # F.pad 第二个参数表示对最后的H and W 进行 左右上下的维度扩展，如 5*3 -> 12(5+3+4) * 6(3+1+2) (pad=(1,2,3,4))

def transVII_kxk_1x1(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k1, k2.permute(1, 0, 2, 3))      # input,weight
        # k1 -> C*D*3*3 (D(output) is arbitrary,s C种通道为D的1*1卷积核),原本是D*C*3*3(C is input,D is output)的卷积运算
        # k2 -> D*E*1*1
        # 由于Transfer 1是 1*1卷积只能通过通道线性组合(channel-wise linear combination but no spatial aggregation)
        # 线性重组k*K卷积核的参数然后合并到 k*k conv中
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        # b_hat = ΣDKK bd * F^2
    return k, b_hat + b2