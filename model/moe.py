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
        gates = self.gate(x)  # 计算每个token的门控值
        weights, indices = torch.topk(gates, self.top_k, dim=-1)  # 获取top_k专家
        weights = F.softmax(weights, dim=-1)  # 对top_k门控值做softmax归一化

        results = torch.zeros_like(x)  # 初始化结果
        # 对每个专家进行加权计算
        for i in range(self.top_k):
            expert_idx = indices[..., i]  # 获取top_k中第i个专家的索引
            expert_mask = F.one_hot(expert_idx, num_classes=len(self.experts))  # 创建one-hot掩码
            expert_output = torch.stack([e(x) for e in self.experts], dim=2)  # 获取每个专家的输出
            expert_output = (expert_output * expert_mask.unsqueeze(-1)).sum(dim=2)  # 选择指定专家的输出
            results += expert_output * weights[..., i].unsqueeze(-1)  # 对输出加权

        return results