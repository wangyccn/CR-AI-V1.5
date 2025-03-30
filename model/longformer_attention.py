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
        # 使用更高效的线性层初始化
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.window_size = 256
        # 添加相对位置偏置
        self.rel_pos_bias = nn.Parameter(torch.randn(1, num_heads, 2 * window_size - 1))
        
    def forward(self, x):
        B, T, D = x.size()
        # 使用更高效的内存布局
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 分块计算注意力
        output = torch.zeros_like(x)
        for i in range(0, T, self.window_size):
            start, end = i, min(i + self.window_size, T)
            # 添加相对位置偏置
            attn = (q[:, :, start:end] @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            attn += self._get_rel_pos_bias(start, end)
            attn = attn.softmax(dim=-1)
            output[:, start:end] = (attn @ v).transpose(1, 2).reshape(B, end-start, D)
        return output