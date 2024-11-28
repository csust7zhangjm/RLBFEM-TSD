import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from models.fcos import FCOS  # 这里替换为你实际导入Fcos模型的方式

# 创建模型实例并将其移动到指定设备（GPU如果可用，否则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCOS(num_classes=3).to(device)  # 根据你的Fcos模型需求设置类别数量等参数

# 定义输入形状，根据Fcos模型实际输入要求来设置
input_shape = (3, 608, 608)

# 计算参数量和计算量
macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                         print_per_layer_stat=True, verbose=False)

print('Computational complexity: ', macs)
print('Number of parameters: ', params)