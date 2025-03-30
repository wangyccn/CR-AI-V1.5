"""混合专家(MoE)模块实现

实现基于门控机制的混合专家系统，支持动态专家选择和负载均衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(2, num_experts)
        
        # 更高效的门控计算
        self.gate = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, num_experts)
        )
        
        # 共享专家第一层权重减少参数
        shared_ff1 = nn.Linear(dim, dim * 4)
        self.experts = nn.ModuleList([
            nn.Sequential(
                shared_ff1,  # 共享第一层
                nn.SiLU(),
                nn.Linear(dim * 4, dim)  # 独立第二层
            ) for _ in range(num_experts)
        ])
        
        # 使用梯度检查点节省内存
        self.expert_checkpoint = torch.utils.checkpoint.checkpoint

    def forward(self, x):
        # 更稳定的门控计算
        gates = self.gate(x.detach())  # 分离梯度流
        
        # 使用top-k专家
        weights, indices = torch.topk(gates, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # 使用梯度检查点
        expert_outputs = []
        for idx in indices.unique():
            expert_out = self.expert_checkpoint(self.experts[idx], x)
            expert_outputs.append(expert_out)
        
        # 组合专家输出
        output = torch.zeros_like(x)
        for i, (w, idx) in enumerate(zip(weights, indices)):
            output[i] = expert_outputs[idx] * w.unsqueeze(-1)
            
        return output