import torch.nn as nn
from model.moe import MoE

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoE(dim, num_experts)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # CNN处理：卷积操作
        x = x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
        x = self.conv(x)  # 进行卷积操作
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)

        # Transformer处理
        attn_out, _ = self.attn(x, x, x)  # 注意力计算
        x = self.norm1(x + attn_out)  # 残差连接 + LayerNorm
        moe_out = self.moe(x)  # MoE模块计算
        x = self.norm2(x + moe_out)  # 残差连接 + LayerNorm
        return x


class MegaLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)  # 嵌入层
        self.layers = nn.ModuleList([  # 堆叠Transformer层
            TransformerBlock(config.dim, config.num_heads, config.num_experts)
            for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.dim, config.vocab_size)  # 输出层

    def forward(self, x):
        x = self.embed(x)  # 嵌入层计算
        for layer in self.layers:  # Transformer层堆叠
            x = layer(x)
        return self.head(x)  # 输出预测