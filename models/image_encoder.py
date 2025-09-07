import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ImageEncoder(nn.Module):
    """
    使用简化版 ResNet 对气体图像数据进行编码
    输入形状: [B, 9, 48, 48]，输出: [B, L, D]，其中 L=1，D=64
    """
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Sequential(nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()),
            nn.Sequential(*resnet_block(32, 32, 1, first_block=True)),
            nn.Sequential(*resnet_block(32, 64, 1)),
            nn.Sequential(*resnet_block(64, 128, 1)),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

    def forward(self, x):  # x: [B, 9, 48, 48]
        x = self.encoder(x)  # [B, 128, 1, 1]
        x = self.fc(x)       # [B, 64]
        return x.unsqueeze(1)  # 加维度 [B, 1, 64] for attention input
