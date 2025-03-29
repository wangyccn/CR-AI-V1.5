import torch.nn as nn

class LongformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.window_size = 256  # 固定窗口大小

    def forward(self, x):
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