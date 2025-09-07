import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, precision_recall_curve
import glob
import os
from pathlib import Path
from sklearn import metrics
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm

#创建文件路径列表
#root_path = 'path/to/your/data/directory'
root_path = 'E:/DataSet/gas+sensor+arrays+in+open+sampling+settings'
paths = [os.path.join(root, subfolder) for root, _, subfolders in os.walk(root_path) for subfolder in subfolders]

#创建目标气体列表['Acetaldehyde_500','Acetone_2500','Ammonia_10000','Benzene_200','Butanol_100','CO_1000','Ethylene_500','Methane_1000','Methanol_200','Toluene_200']
gas_list = [folder for folder in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, folder))]

#创建创建结果文件夹  'E:\\DataSet\\Image1results'
results_dir = os.path.abspath(os.path.join(root_path, os.pardir, 'ToImageresults'))
os.makedirs(results_dir, exist_ok=True)

print(paths[0:1])
print(paths[0:10])
print(len(paths))

# 初始化一个空列表用于存储每个样本的 72x72 矩阵
# 初始化一个空列表用于存储每个样本标签数据
all_samples = []
all_labels=[]

#下采样函数：data为需下采样序列数据     target_length为下采样维度(采用数据填补法和分区间平均法)
def downsample_with_padding(data, target_length):
    segment_size = len(data) // target_length
    remainder = len(data) % target_length   
    # 检查是否需要填充
    if remainder > 0:
        # 计算需要填充的数据量，并填充数据末尾
        padding_length = target_length - remainder
        data = np.concatenate([data, np.full(padding_length, data[-1])])    
    # 将填充后的数据进行分段并取均值
    return np.mean(data.reshape(target_length, -1), axis=1)

#处理原始数据文件,下采样至72维(区间平均法，不够用最后一个数据进行补全)
for file in paths:
    raw_data = pd.read_csv(file, delimiter='\t', header=None, usecols=[filecol for filecol in range(1, 92)],engine='python', dtype='float32')
    
    # 初始化新数据数组，用于存储72行（传感器数据）
    new_data = np.zeros((72, 72), dtype='float32')
    # 获取当前气体的编号（根据文件路径推断）从0开始
    gas_number = gas_list.index(file.split(os.path.sep)[-3])
    all_labels.append(gas_number)
    # 逐个模块和传感器进行处理并下采样
    row_index = 0  # 记录在 new_data 中的行位置    
    for vertical_location in range(9):
        col = 12 + 9 * vertical_location
        for sensor_number in range(8):
            sensor_data = raw_data[col + sensor_number].to_numpy()
            downsample_sensor_data = downsample_with_padding(sensor_data,72)
            # 将下采样后的传感器数据放入new_data的对应行
            new_data[row_index, :] = downsample_sensor_data
            row_index += 1  # 更新行位置       
    # 确保 new_data 是 72x72 的矩阵
    assert new_data.shape == (72, 72), "new_data 不是 72x72 的矩阵"
    all_samples.append(new_data.T)

#处理后的数据processed_data72为(样本个数,传感器个数，传感器特征数)
#处理后的标签all_labels
processed_data72 = np.stack(all_samples)
labels = all_labels
print("************************************数据为************************************")
print(processed_data72)
print(processed_data72.shape)
print(type(processed_data72))
print("************************************标签************************************")
print(labels)
print(len(labels))

#将气体数据处理为Imagedata1的数据和标签保存到processed_Image1data.npz文件中
# (15783, 72, 72) numpy    15783 []列表
processed_data72[11583][:,32]=334.3901701
processed_data72[11584][:,32]=403.1842758
np.savez('E:/DataSet/ToImageresults/processed_Imagedata1.npz', data=processed_data72, labels=labels)

#数据预处理为ImageData2的形式

# 初始化一个空列表用于存储每个样本的 72x72 矩阵
# 初始化一个空列表用于存储每个样本标签数据
ImaData2_all_samples = []
ImaData2_all_labels=[]

# 处理原始数据文件,并且可视化处理进度*
for file in tqdm(paths, desc="Processing files", unit="file"):
    raw_data = pd.read_csv(file, delimiter='\t', header=None, usecols=[filecol for filecol in range(1, 92)], engine='python', dtype='float32')
    # 初始化新数据数组，用于存储72行（传感器数据）
    new_data = np.zeros((72, 72), dtype='float32')
    # 获取当前气体的编号（根据文件路径推断）
    gas_number = gas_list.index(file.split(os.path.sep)[-3])
    ImaData2_all_labels.append(gas_number)
    
    # 逐个模块和传感器进行处理下采样
    
    for vertical_location in range(9):
        col = 12 + 9 * vertical_location
        for sensor_number in range(8):
            sensor_data = raw_data[col + sensor_number].to_numpy()
            downsample_sensor_data = downsample_with_padding(sensor_data, 72).reshape(8,9)
            # 将下采样后的传感器数据放入new_data的对应行
            new_data[vertical_location*8:(vertical_location+1)*8, sensor_number*9:(sensor_number+1)*9] = downsample_sensor_data

    # 确保 new_data 是 72x72 的矩阵
    assert new_data.shape == (72, 72), "new_data 不是 72x72 的矩阵"
    ImaData2_all_samples.append(new_data)

processed_ImageData2 = np.stack(ImaData2_all_samples)
print("************************************数据为************************************")
print(processed_ImageData2)
print(processed_ImageData2.shape)
print(type(processed_ImageData2))
print("************************************标签************************************")
print(ImaData2_all_labels)
print(len(ImaData2_all_labels))

first_image = processed_ImageData2[0].squeeze()
# 绘制灰度图
plt.imshow(first_image, cmap='gray')
plt.colorbar()  # 显示颜色条，便于查看灰度值
plt.title("First Sample in Grayscale")
plt.axis('off')  # 隐藏坐标轴
plt.show()

Second_image = processed_ImageData2[1].squeeze()
# 绘制灰度图
plt.imshow(Second_image, cmap='gray')
plt.colorbar()  # 显示颜色条，便于查看灰度值
plt.title("Second Sample in Grayscale")
plt.axis('off')  # 隐藏坐标轴
plt.show()

np.savez('E:/DataSet/ToImageresults/processed_Imagedata2.npz', data=processed_ImageData2, labels=ImaData2_all_labels)

#数据预处理为ImageData3的形式(数据下采样到256维)

# 初始化一个空列表用于存储每个样本的 9x48x48 矩阵
# 初始化一个空列表用于存储每个样本标签数据
ImaData3_all_samples = []
ImaData3_all_labels=[]

# 处理原始数据文件，按照 ImageData3 的方式组织成多个通道
for file in tqdm(paths, desc="Processing files", unit="file"):
    raw_data = pd.read_csv(file, delimiter='\t', header=None, usecols=[filecol for filecol in range(1, 92)], engine='python', dtype='float32')
    
    # 初始化新数据数组，用于存储 9x48x48 的图像数据，每个传感器板块为一个通道
    sample_data = np.zeros((9, 48, 48), dtype='float32')  # 9个通道，每个通道 48x48

    
    gas_number = gas_list.index(file.split(os.path.sep)[-3]) # 气体编号，实际应根据文件路径推断
    ImaData3_all_labels.append(gas_number)
    
    # 对每个模块的8个传感器数据进行处理
    for board in range(9):  # 9个板块
        channel_image = np.zeros((48, 48), dtype='float32')
        for sensor_number in range(8): # 8个传感器
            col = 12 + 9 * board + sensor_number
            sensor_data = raw_data[col].to_numpy()
            downsample_sensor_data = downsample_with_padding(sensor_data, 256).reshape(16, 16)
            
            # 放置到3x3网格中的正确位置
            row_start = (sensor_number // 3) * 16
            col_start = (sensor_number % 3) * 16
            channel_image[row_start:row_start+16, col_start:col_start+16] = downsample_sensor_data
        
        # 将板块图像填充到sample_data的对应通道中
        sample_data[board] = channel_image
    
    assert sample_data.shape == (9, 48, 48), "样本数据形状不正确，应为 (9, 48, 48)"
    ImaData3_all_samples.append(sample_data)

processed_ImageData3 = np.stack(ImaData3_all_samples)
print("************************************数据为************************************")
print(processed_ImageData3)
print(processed_ImageData3.shape)
print(type(processed_ImageData3))

print("************************************标签************************************")
print(ImaData3_all_labels)
print(len(ImaData3_all_labels))

np.savez('E:/DataSet/ToImageresults/processed_Imagedata3.npz', data=processed_ImageData3, labels=ImaData3_all_labels)

