import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
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
    model: ModelConfig
    training: TrainingConfig


def load_config(path):
    with open(path) as f:
        config_data = json.load(f)

    # 构建嵌套配置对象
    model_config = ModelConfig(**config_data['model'])
    training_config = TrainingConfig(**config_data['training'])

    return Config(
        model=model_config,
        training=training_config
    )