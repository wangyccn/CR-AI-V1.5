import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([  # 每个专家由两个线性层组成
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)  # Gate网络用于分配专家
        self.top_k = top_k  # 使用top_k个专家
        self.dim = dim  # 输入维度

    def forward(self, x):
        # 计算门控值并进行top_k选择
        gates = self.gate(x)  # 获取每个token的门控值
        weights, indices = torch.topk(gates, self.top_k, dim=-1)  # 获取top_k专家
        weights = F.softmax(weights, dim=-1)  # 对top_k门控值做softmax归一化

        # 预先计算所有专家的输出
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)  # [batch_size, seq_len, num_experts, dim]

        # 使用批量操作进行专家选择和加权计算
        expert_mask = torch.gather(expert_outputs, dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, self.dim))  # 选择top_k专家
        weighted_expert_outputs = expert_mask * weights.unsqueeze(-1)  # 对输出加权

        return weighted_expert_outputs.sum(dim=2)  # 聚合结果