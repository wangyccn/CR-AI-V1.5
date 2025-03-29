import torch
from tokenizers import Tokenizer
from typing import List, Optional, Union
from utils.config import load_config
from model.architecture import MegaLLM
from PIL import Image
from torchvision import transforms

class ModelLoader:
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        
        # 添加图像预处理
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_config(self, config_path: str) -> dict:
        """加载模型配置"""
        try:
            config = load_config(config_path)
            return config.__dict__ if hasattr(config, '__dict__') else config
        except Exception as e:
            raise RuntimeError(f"加载配置失败：{str(e)}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        try:
            model_config = self.config.get('model')
            if not model_config:
                raise ValueError("配置中缺少model配置")
                
            model = MegaLLM(model_config)
            # 添加更详细的加载错误信息
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"无法加载模型文件 {model_path}: {str(e)}")
            
            # 添加状态字典检查
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise ValueError("检查点文件中缺少model_state_dict")
            else:
                model.load_state_dict(checkpoint)
                
            model = model.to(self.device, dtype=self.dtype)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败：{str(e)}")

    def process_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """处理输入图像"""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    return image
                else:
                    raise ValueError("输入张量维度不正确")
                    
            if isinstance(image, Image.Image):
                image = self.image_processor(image)
                
            return image.unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"图像处理失败：{str(e)}")

    def predict(self, tokenizer: Tokenizer, query: str, 
               image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
               history: List[str] = None, 
               temperature: float = 1.0, 
               top_p: float = 0.95) -> str:
        """生成多模态响应"""
        try:
            history = history or []
            
            # 处理输入
            input_text = " ".join(history + [query])
            input_ids = torch.tensor(tokenizer.encode(input_text).ids).unsqueeze(0).to(self.device)
            
            # 处理图像（如果有）
            image_tensor = self.process_image(image) if image is not None else None
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_length=self.config.get('max_length', 100),
                    temperature=temperature,
                    top_p=top_p
                )
                
                response = tokenizer.decode(output_ids[0].tolist())
                return response.split(query)[-1].strip()
                
        except Exception as e:
            raise RuntimeError(f"生成失败：{str(e)}")
