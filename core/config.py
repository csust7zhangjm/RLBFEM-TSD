from core.filters import *
import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='')
# parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use DIP Module')


args = parser.parse_args()
__C                             = edict()
cfg                             = __C


#------------------------------------------------滤波器参数------------------------------------------------------------------
cfg.filters = [
    # GammaFilter,
    ExposureFilter, ToneFilter, LumFilter
]

cfg.num_filter_parameters = 10  #预测参数个数


#------------------------------------------------------------------CNN-PP网络参数------------------------------------------------
cfg.cnnpp_channel = 16  #输入channel


#-----------------------------------------------------------DoubleHead 预测头参数------------------------------------------------
cfg.head_channel = 256  #输入channel
cfg.head_channel_2 = 1024
cfg.num_convs = 3  #设置预测头的ResidualBottleNeck个数（3）