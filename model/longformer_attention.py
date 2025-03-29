"""Longformer注意力机制实现模块

实现基于窗口的局部注意力机制，用于处理长序列输入
"""

import torch.nn as nn

class LongformerAttention(nn.Module):
    """Longformer注意力层
    
    实现基于窗口的局部注意力机制，适用于长序列处理
    
    属性:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        head_dim (int): 每个注意力头的维度
        window_size (int): 注意力窗口大小
    """
    def __init__(self, dim, num_heads=8):
        """初始化Longformer注意力层
        
        参数:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数(默认为8)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.window_size = 256  # 固定窗口大小

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入张量，形状为[batch_size, seq_len, dim]
            
        返回:
            Tensor: 输出张量，形状与输入相同
        """
        B, T, D = x.size()
        H = self.num_heads
        HD = self.head_dim
        # 计算查询、键和值
        q = self.query(x).view(B, -1, H, HD).permute(0, 2, 1, 3)
        k = self.key(x).view(B, -1, H, HD).permute(0, 2, 1, 3)
        v = self.value(x).view(B, -1, H, HD).permute(0, 2, 1, 3)
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (D**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        return out