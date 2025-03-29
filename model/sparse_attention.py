import torch.nn as nn

class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, block_size=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
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