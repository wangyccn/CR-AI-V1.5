"""混合专家(MoE)模块实现

实现基于门控机制的混合专家系统，支持动态专家选择和负载均衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    """混合专家层
    
    实现基于门控机制的混合专家系统，支持:
    - 动态专家选择
    - 负载均衡损失
    - 专家丢弃
    
    属性:
        dim (int): 输入/输出特征维度
        num_experts (int): 专家数量
    """
    def __init__(self, dim, num_experts):
        """初始化混合专家层
        
        参数:
            dim (int): 输入/输出特征维度
            num_experts (int): 专家数量
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        # 使用更高效的专家结构
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),  # 专家前馈网络第一层
                nn.SiLU(),                # 激活函数
                nn.Linear(dim * 4, dim)   # 专家前馈网络第二层
            ) for _ in range(num_experts)
        ])
        
        # 添加专家丢弃率
        self.expert_dropout = nn.Dropout(0.1)
        
        # 使用更稳定的初始化
        for expert in self.experts:
            nn.init.orthogonal_(expert[0].weight)  # 正交初始化第一层权重
            nn.init.zeros_(expert[0].bias)         # 零初始化第一层偏置
            nn.init.kaiming_normal_(expert[2].weight)  # Kaiming初始化第二层权重
            nn.init.zeros_(expert[2].bias)             # 零初始化第二层偏置

    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 输入张量，形状为[batch_size, seq_len, dim]
            
        返回:
            Tensor: 输出张量，形状与输入相同
        """
        # 添加门控温度调节
        gates = self.gate(x) / (self.temperature + 1e-6)
        
        # 使用top-k+gumbel softmax
        if self.training:
            gates = F.gumbel_softmax(gates, tau=0.1, hard=True, dim=-1)
        else:
            weights, indices = torch.topk(gates, self.top_k, dim=-1)
            weights = F.softmax(weights, dim=-1)
        
        # 添加专家丢弃
        expert_outputs = torch.stack([self.expert_dropout(e(x)) for e in self.experts], dim=2)
        
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
        """获取负载均衡损失
        
        返回:
            Tensor: 负载均衡损失值
        """
        return self.load_balance_loss