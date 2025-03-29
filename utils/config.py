"""模型和训练配置管理模块

提供以下功能:
1. 模型配置(ModelConfig): 定义模型结构参数
2. 训练配置(TrainingConfig): 定义训练超参数
3. 配置加载(load_config): 从JSON文件加载配置

配置项说明:
- ModelConfig: 包含模型结构、多模态支持等参数
- TrainingConfig: 包含训练过程、优化器、批次大小等参数
- Config: 整合模型和训练配置的容器类

使用方式:
1. 创建JSON配置文件
2. 使用load_config加载配置
3. 通过Config对象访问配置参数
"""

import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型结构配置类
    
    属性:
        vocab_size (int): 词汇表大小
        dim (int): 模型隐藏层维度
        num_layers (int): Transformer层数
        num_heads (int): 注意力头数
        num_experts (int): MoE专家数量
        use_longformer (bool): 是否使用Longformer注意力
        multimodal (bool): 是否支持多模态输入
        image_dim (int): 图像特征维度
        max_seq_len (int): 最大输入序列长度
    """
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_experts: int
    use_longformer: bool = False
    multimodal: bool = False  # 新增多模态支持
    image_dim: int = 512  # 图像特征维度
    max_seq_len: int = 8192  # 最大序列长度

@dataclass 
class TrainingConfig:
    """训练过程配置类
    
    属性:
        batch_size (int): 训练批次大小
        lr (float): 初始学习率
        epochs (int): 训练轮数
        data_path (str): 训练数据路径
        save_dir (str): 模型保存目录
        save_interval (int): 模型保存间隔(步数)
        num_workers (int): 数据加载工作进程数
        min_batch_size (int): 最小动态批次大小
        max_batch_size (int): 最大动态批次大小
        warmup_steps (int): 学习率预热步数
        grad_accum_steps (int): 梯度累积步数
        fp16 (bool): 是否使用混合精度训练
        bf16 (bool): 是否使用bfloat16精度
    """
    batch_size: int
    lr: float
    epochs: int
    data_path: str
    save_dir: str
    save_interval: int
    num_workers: int
    min_batch_size: int = 8  # 最小批次大小
    max_batch_size: int = 128  # 最大批次大小
    warmup_steps: int = 2000  # 学习率预热步数
    grad_accum_steps: int = 4  # 梯度累积步数
    fp16: bool = True  # 混合精度训练
    bf16: bool = False


@dataclass
class Config:
    """整合配置容器类
    
    属性:
        model (ModelConfig): 模型结构配置
        training (TrainingConfig): 训练过程配置
    """
    model: ModelConfig
    training: TrainingConfig


def load_config(path):
    """从JSON文件加载配置
    
    参数:
        path (str): JSON配置文件路径
        
    返回:
        Config: 配置对象
    """
    with open(path) as f:
        config_data = json.load(f)

    # 构建嵌套配置对象
    model_config = ModelConfig(**config_data['model'])
    training_config = TrainingConfig(**config_data['training'])

    return Config(
        model=model_config,
        training=training_config
    )