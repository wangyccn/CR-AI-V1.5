import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def init_distributed():
    """初始化分布式训练环境"""
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


rank, world_size = init_distributed()


class RMSNorm(nn.Module):
    """RMS规范化层"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class ParallelEmbedding(nn.Module):
    """并行嵌入层，支持分布式词汇表分片"""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x):
        if world_size > 1:
            # 创建掩码，标记不在当前进程词汇分区中的token
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # 调整输入索引到本地词汇范围
            x_adjusted = x - self.vocab_start_idx
            x_adjusted[mask] = 0  # 将不在分区中的token索引设为0
            y = F.embedding(x_adjusted, self.weight)
            y[mask] = 0  # 将不在分区中的嵌入设为0
            dist.all_reduce(y)  # 跨进程求和，合并所有嵌入
        else:
            # 单进程模式，直接计算嵌入
            y = F.embedding(x, self.weight)
        return y
