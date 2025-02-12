import torch
from tokenizers import Tokenizer
from typing import List
from utils.config import load_config
from model.architecture import MegaLLM  # 引入 MegaLLM 而不是 TransformerBlock

class ModelLoader:
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        # 保持原始接口，不改变调用方式
        self.device = device
        self.dtype = dtype
        self.config = self._load_config(config_path)  # 使用封装方法加载新的配置
        self.model = self._load_model(model_path)

    def _load_config(self, config_path: str):
        # 加载新配置，返回一个新的Config对象
        return load_config(config_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        # 从配置中获取新的模型配置
        model_config = self.config.model  # 这里的model配置需要适配新的模型结构
        model = MegaLLM(model_config)  # 使用新的 MegaLLM 初始化模型
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device, dtype=self.dtype)
        return model

    def predict(self, tokenizer: Tokenizer, query: str, history: List[str], temperature: float = 1.0, top_p: float = 0.95) -> str:
        with torch.no_grad():
            # 保持原接口，直接返回生成的响应
            for response, _ in self.model.stream_generate(
                    query=query,
                    history=history,
                    temperature=temperature,
                    top_p=top_p
            ):
                return response
