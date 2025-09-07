import torch
import torch.nn as nn

class EnvironmentEncoder(nn.Module):
    """
    MLP编码器：将温度、湿度、风速等环境信息编码为中间特征 E_g
    输入维度：如3（温度、湿度、风速）
    输出维度：与主模态对齐，例如 64
    """
    def __init__(self, input_dim=3, hidden_dims=[32, 64]):
        super(EnvironmentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )

    def forward(self, x):
        return self.encoder(x)
