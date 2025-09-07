import torch
import torch.nn as nn

class CMAC(nn.Module):
    """
    Cross-Modal Attention Compensation
    输入：主模态特征 F_g，环境模态特征 E_g
    输出：最终融合特征 Y_g
    """
    def __init__(self, dim):
        super(CMAC, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F_g, E_g):
        # F_g, E_g: [B, D]
        Q = self.query_proj(F_g).unsqueeze(1)  # [B, 1, D]
        K = self.key_proj(E_g).unsqueeze(1)    # [B, 1, D]
        V = self.value_proj(E_g).unsqueeze(1)  # [B, 1, D]

        attn_weights = self.softmax(torch.bmm(Q, K.transpose(1, 2)))  # [B, 1, 1]
        attended = torch.bmm(attn_weights, V).squeeze(1)  # [B, D]

        Y_g = F_g + attended
        return Y_g
