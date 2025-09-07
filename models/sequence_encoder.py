import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """
    Transformer 序列编码器
    输入：形状 [B, seq_len, input_dim] 的气体传感器时间序列
    输出：形状 [B, 1, D] 的特征向量序列
    """
    def __init__(self, input_dim, seq_len, num_classes, d_model=128, num_heads=8, num_layers=4, dropout=0.1, feedforward_dim=512):
        super(TransformerClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.input_embedding = nn.Linear(4 * input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=feedforward_dim, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, T, C]
        batch_size, seq_len, input_dim = x.shape
        x = x.view(batch_size * seq_len, 1, input_dim)
        x = self.conv1d(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.input_embedding(x) + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = self.global_pooling(x.permute(0, 2, 1)).squeeze(-1)  # [B, d_model]
        return x.unsqueeze(1)  # [B, 1, d_model]
