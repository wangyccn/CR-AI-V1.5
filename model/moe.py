import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        # 修改专家网络结构，添加激活函数和更合理的参数初始化
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        
        # 添加参数初始化（修复缩进问题）
        for expert in self.experts:
            for layer in expert.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.normal_(layer.bias, std=1e-6)

        # 路由网络定义（移出循环）
        self.gate = nn.Linear(dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 新增温度参数
        self.top_k = 2  # 新增top_k参数

    def forward(self, x):
        # 门控计算
        gates = self.gate(x)
        gates = gates / self.temperature
        weights, indices = torch.topk(gates, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # 计算负载均衡损失
        expert_counts = torch.sum(torch.zeros_like(indices).scatter(1, indices, 1), dim=0)
        expert_counts = (expert_counts / expert_counts.sum()).square().sum()
        self.load_balance_loss = expert_counts
        # 专家网络输出
        all_expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        expert_outputs = torch.gather(all_expert_outputs, 2, indices.unsqueeze(-1).expand(*all_expert_outputs.size()[:3], self.dim))
        outputs = (expert_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return outputs

    def get_loss(self):
        return self.load_balance_loss