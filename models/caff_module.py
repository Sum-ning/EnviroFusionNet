import torch
import torch.nn as nn
import torch.nn.functional as F

class CAFF(nn.Module):
    """
    Cross-Attention Fusion Framework
    输入：图像空间特征 I_g 和 序列时间特征 S_g
    输出：融合后的主模态特征 F_g
    """
    def __init__(self, feature_dim, num_heads=4):
        super(CAFF, self).__init__()
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

    def forward(self, I_g, S_g):
        # I_g: [B, L1, D], S_g: [B, L2, D]
        F_1, _ = self.cross_attn_1(I_g, S_g, S_g)
        F_2, _ = self.cross_attn_2(S_g, I_g, I_g)
        F_g = F_1 + F_2
        return F_g.mean(dim=1)  # [B, D]
