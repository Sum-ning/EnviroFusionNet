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
results_dir = os.path.abspath(os.path.join(root_path, os.pardir, 'ToSequenceresults'))
os.makedirs(results_dir, exist_ok=True)

print(paths[0:1])
print(paths[0:10])
print(len(paths))

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

#数据预处理为Sequence1的形式: seq_len时间步为72 input_dim数据维度为传感器数量72
#时间步1: [传感器1, 传感器2, ..., 传感器72]
#时间步2: [传感器1, 传感器2, ..., 传感器72]
#...
#时间步260: [传感器1, 传感器2, ..., 传感器72]

# 初始化一个空列表用于存储每个样本的 260*72 矩阵
# 初始化一个空列表用于存储每个样本标签数据
SeqData1_all_samples = []
SeqData1_all_labels=[]

# 处理原始数据文件,并且可视化处理进度*
for file in tqdm(paths, desc="Processing files", unit="file"):
    raw_data = pd.read_csv(file, delimiter='\t', header=None, usecols=[filecol for filecol in range(1, 92)], engine='python', dtype='float32')
    # 初始化新数据数组，用于存储72行（传感器数据）
    new_data = np.zeros((260, 72), dtype='float32')
    # 获取当前气体的编号（根据文件路径推断）
    gas_number = gas_list.index(file.split(os.path.sep)[-3])
    
    SeqData1_all_labels.append(gas_number)
    
    # 逐个模块和传感器进行处理下采样
    col_index = 0 #记录在 new_data 中的列位置
    for vertical_location in range(9):
        col = 12 + 9 * vertical_location
        for sensor_number in range(8):
            sensor_data = raw_data[col + sensor_number].to_numpy()
            downsample_sensor_data = downsample_with_padding(sensor_data,260)
            new_data[:,col_index] = downsample_sensor_data
            col_index += 1
            # 将下采样后的传感器数据放入new_data的对应列
            
    # 确保 new_data 是 72x72 的矩阵
    assert new_data.shape == (260, 72), "new_data 是不是 260x72 的矩阵"
    SeqData1_all_samples.append(new_data)

processed_SeqData1 = np.stack(SeqData1_all_samples)
print("************************************数据************************************")
print(processed_SeqData1)
print(processed_SeqData1.shape)
print(type(processed_SeqData1))
print("************************************标签************************************")
print(SeqData1_all_labels)
print(len(SeqData1_all_labels))

processed_SeqData1[5713][9,32]=355
processed_SeqData1[5713][10,32]=361

np.savez('E:/DataSet/ToSequenceresults/processed_SeqData1.npz', data=processed_SeqData1, labels=SeqData1_all_labels)

