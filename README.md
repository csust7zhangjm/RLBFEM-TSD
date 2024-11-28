# An Improved Anchor-Free One-stage Traffic Sign Detector with Reparameterized Long-range Bi-directional Feature Enhancement

# train

## Environment configuration

1. cuda>=11.6
2. For installation of other libraries, please refer to **requirements.txt**

## Training steps

1. Configuration modification
   1. Enter the **train.py** file and modify the relevant configuration according to the comments, paying special attention to the modification of the **classs_path** field value
   2. Path modification of pretrain weight files
2. For the generation of path documents for the train and test sets, please refer to the **voc_annotation** file
3. Modify the corresponding train and test set document paths in **train.py**
4. The train results are saved in the **logs** folder

# Test

1. Modify the field value of **model_path** in **focs.py** to the trained weight file
2. Enter the **get_map. py** file and check if the path of the dataset images is correct
3. Run the **get_map. py** file and save the test results in the **map_out** folder
4. The testing of parameter and computational complexity is conducted in **complexity_and_parameters.py**

# Datasets

CCTSDB 2021：[GitHub - csust7zhangjm/CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021)

GTSDB：[GTSDB - Dataset Ninja](https://datasetninja.com/gtsdb)

TT100K：[TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)


# 训练

## 环境配置

1. cuda>=11.6
2. 其他库安装参见**requirements.txt**

## 训练步骤

1. 配置修改
   1. 进入train.py文件，按照注释修改相关配置，特别注意**classes_path**z字段值的修改
   2. 预训练权重文件的路径修改
2. 训练集和测试集路径文档的生成，参见**voc_annotation**文件
3. 修改train.py里面对应的训练集和测试集文档路径

4. 训练结果保存在**logs**文件夹下

# 测试

1. 修改**focs.py**中的**model_path**字段值为训练好的权重文件
2. 进入**get_map.py**文件，检查数据集图片的路径是否正确
3. 运行**get_map.py**文件，测试结果保存在**map_out**文件夹下

4. 参数量和计算量的测试在**complexity_and_parameters.py**

# 数据集

CCTSDB 2021：[GitHub - csust7zhangjm/CCTSDB2021](https://github.com/csust7zhangjm/CCTSDB2021)

GTSDB：[GTSDB - Dataset Ninja](https://datasetninja.com/gtsdb)

TT100K：[TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
