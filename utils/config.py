import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_experts: int


@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    data_path: str
    save_dir: str
    save_interval: int


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