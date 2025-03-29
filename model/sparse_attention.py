"""稀疏注意力机制实现模块

实现基于分块的稀疏注意力机制，用于处理长序列输入
"""

import torch.nn as nn

class SparseAttention(nn.Module):
    """稀疏注意力层
    
    通过分块处理实现高效的长序列注意力计算
    
    属性:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数
        block_size (int): 分块大小
    """
    def __init__(self, dim, num_heads=8, block_size=16):
        """初始化稀疏注意力层
        
        参数:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数(默认为8)
            block_size (int): 分块大小(默认为16)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入张量，形状为[batch_size, seq_len, dim]
            
        返回:
            Tensor: 输出张量，形状与输入相同
        """
        B, T, D = x.size()
        H = self.num_heads
        BS = self.block_size
        # 分块处理
        x = x.view(B, -1, BS, D).permute(0, 2, 1, 3)
        q = self.query(x).view(B, BS, -1, H, D//H).permute(0, 1, 3, 2, 4)
        k = self.key(x).view(B, BS, -1, H, D//H).permute(0, 1, 3, 4, 2)
        v = self.value(x).view(B, BS, -1, H, D//H).permute(0, 1, 3, 2, 4)
        # 注意力计算
        attn = (q @ k) * (D**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, BS, -1, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return out