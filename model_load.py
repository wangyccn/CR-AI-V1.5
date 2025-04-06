"""MegaLLM 模型加载和推理模块

该模块提供:
- 模型加载与量化
- 单样本和批量推理
- 图像预处理
- 性能优化

使用示例:
    loader = ModelLoader("model.pth", "config.json")
    response = loader.predict(tokenizer, "你好")
"""

import torch
from tokenizers import Tokenizer
from typing import List, Optional, Union
from utils.config import load_config
from model.architecture import MegaLLM
from PIL import Image
from utils.config import load_config
from model.architecture import MegaLLM
from torchvision import transforms
import logging
import time
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelLoader")


__all__ = ['ModelLoader']  # 添加这行确保类被正确导出

class ModelLoader:
    def __init__(self, config_path: str, device: str = 'cuda',
                 dtype: torch.dtype = torch.float16, quantize: bool = False):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型权重路径(弃用并移除)
            config_path: 配置文件路径
            device: 运行设备，默认为'cuda'
            dtype: 模型精度，默认为float16
            quantize: 是否量化模型，默认为False
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.dtype = dtype
        self.quantize = quantize
        self.config = self._load_config(config_path)
        model_config = self.config.get('model')
        model_path = getattr(model_config, "model_path", "longformer_pretrained.pth")
        base_path = os.path.dirname(os.path.abspath(__file__))
        # 定义模型和配置文件的相对路径
        model_path = os.path.join(base_path, model_path)

        # 添加缓存
        self._cache = {}

        # 添加图像预处理
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])  # CLIP标准归一化
        ])

        # 添加8-bit量化选项
        self.quantize_8bit = quantize

        # 加载模型
        logger.info(f"正在加载模型: {model_path}")
        start_time = time.time()
        self.model = self._load_model(model_path)
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

    def _load_config(self, config_path: str) -> dict:
        """加载模型配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 配置字典
        """
        try:
            logger.info(f"正在加载配置: {config_path}")
            config = load_config(config_path)
            # 确保返回字典类型
            if hasattr(config, '__dict__'):
                return vars(config)  # 使用vars()替代__dict__更规范
            elif isinstance(config, dict):
                return config
            else:
                raise ValueError("配置必须是字典或具有__dict__属性的对象")
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise RuntimeError(f"加载配置失败：{str(e)}")
    def _load_model(self, model_path: str) -> torch.nn.Module:
        try:
            model_config = self.config.get('model')
            if not model_config:
                raise ValueError("配置中缺少model配置")
            if callable(model_config):
                raise ValueError("model配置应该是字典或对象而不是函数")
        
            # 确保model_config是有效的配置对象/字典
            if isinstance(model_config, dict):
                # 如果是字典，检查必要参数
                required_keys = ['dim', 'num_heads', 'num_layers']
                for key in required_keys:
                    if key not in model_config:
                        raise ValueError(f"model配置缺少必要参数: {key}")
            elif not hasattr(model_config, '__dict__'):
                raise ValueError("model配置必须是字典或具有__dict__属性的对象")
        
            # 确保MegaLLM被正确实例化
            try:
                # 修改：确保MegaLLM是一个类而不是函数
                if callable(MegaLLM) and not isinstance(MegaLLM, type):
                    raise TypeError("MegaLLM应该是一个类，但它是一个函数")
                
                model = MegaLLM(model_config)
                
                # 添加类型检查
                if not isinstance(model, torch.nn.Module):
                    raise TypeError(f"模型必须是torch.nn.Module的实例，而不是{type(model)}")
                    
                # 确保模型有eval方法
                if not hasattr(model, 'eval') or not callable(getattr(model, 'eval')):
                    raise AttributeError("模型缺少eval方法")
                    
                # 检查model是否为有效的PyTorch模型
                if not isinstance(model, torch.nn.Module):
                    raise TypeError(f"模型实例化失败，得到的是{type(model)}而不是torch.nn.Module")
            except Exception as e:
                logger.error(f"模型实例化失败: {str(e)}")
                raise ValueError(f"无法创建MegaLLM模型实例: {str(e)}")

            # 添加更详细的加载错误信息
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")

                # 修改1: 添加安全全局变量
                from utils.config import Config
                with torch.serialization.safe_globals([Config]):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception as e:
                # 修改2: 如果安全加载失败，尝试非安全加载（仅当信任模型来源时）
                logger.warning(f"安全加载失败，尝试非安全加载: {str(e)}")
                # 直接使用非安全加载
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # 添加状态字典检查
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise ValueError("检查点文件中缺少model_state_dict")
            else:
                model.load_state_dict(checkpoint)

            # 模型优化 - 将多个优化步骤合并
            if self.device == 'cuda':
                # 量化模型以减少内存使用并提高推理速度
                if self.quantize:
                    logger.info("正在进行模型量化...")
                    try:
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        logger.info("模型量化完成")
                    except Exception as e:
                        logger.warning(f"模型量化失败: {str(e)}，将使用原始模型")

                # 8-bit量化支持
                if self.quantize_8bit:
                    try:
                        import bitsandbytes as bnb
                        logger.info("正在进行8-bit量化...")
                        model = bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                            model, 'weight', {'optim_bits': 8}
                        )
                        logger.info("8-bit量化完成")
                    except ImportError:
                        logger.warning("bitsandbytes未安装，跳过8-bit量化")
                    except Exception as e:
                        logger.warning(f"8-bit量化失败: {str(e)}")

                # 使用torch.compile加速模型（仅执行一次）
                if hasattr(torch, 'compile'):
                    logger.info("正在使用torch.compile优化模型...")
                    try:
                        # 修复：移除mode参数，只保留options
                        model = torch.compile(
                            model,
                            # 移除mode参数
                            fullgraph=False,
                            dynamic=True,
                            backend='inductor',
                            options={
                                'triton.cudagraphs': True,
                                'shape_padding': True
                            }
                        )
                        logger.info("模型编译完成")
                    except Exception as e:
                        logger.warning(f"模型编译失败: {str(e)}")

            # 确保模型处于评估模式 (修复eval调用)
            if not hasattr(model, 'eval') or not callable(getattr(model, 'eval')):
                raise AttributeError(f"模型对象没有eval方法，类型为: {type(model)}")
            
            model = model.eval()

            # 预热模型
            self._warmup_model(model)

            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise RuntimeError(f"加载模型失败：{str(e)}")

    def _warmup_model(self, model: torch.nn.Module) -> None:
        """预热模型以优化初次推理性能

        Args:
            model: 要预热的模型
        """
        try:
            logger.info("正在预热模型...")
            # 创建一个小的随机输入进行预热
            dummy_input = torch.randint(0, 100, (1, 10)).to(self.device)
            
            # 检查模型是否有embedding_dim属性
            vocab_size = 100
            if hasattr(model, 'embedding_dim'):
                vocab_size = model.embedding_dim
            elif hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                vocab_size = model.config.vocab_size
            
            dummy_input = torch.randint(0, vocab_size, (1, 10)).to(self.device)
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)

            # 使用torch.cuda.synchronize确保GPU操作完成
            with torch.no_grad():
                for _ in range(3):  # 预热几次
                    _ = model.generate(input_ids=dummy_input, images=dummy_image, max_length=20)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()

            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {str(e)}")

    def process_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """处理输入图像为模型可接受的张量

        Args:
            image: 输入图像，可以是路径、PIL图像或张量

        Returns:
            torch.Tensor: 处理后的图像张量

        Raises:
            RuntimeError: 如果处理失败
        """
        try:
            # 检查缓存
            if isinstance(image, str) and image in self._cache:
                return self._cache[image]

            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    return image.to(self.device)
                else:
                    raise ValueError("输入张量维度不正确，应为[B, C, H, W]")

            if isinstance(image, Image.Image):
                image = self.image_processor(image)

            # 确保图像是张量类型并添加batch维度
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image)
            processed_image = image.unsqueeze(0).to(self.device)

            # 缓存结果
            if isinstance(image, str):
                self._cache[image] = processed_image

            return processed_image
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            raise RuntimeError(f"图像处理失败：{str(e)}")

    def predict(self, tokenizer: Tokenizer, text: str, image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
               history: Optional[List[str]] = None, temperature: float = 1.0, top_p: float = 0.95,
               max_length: Optional[int] = None, num_beams: int = 1):
        """
        执行推理

        Args:
            tokenizer: 分词器实例
            text: 输入文本
            image: 可选输入图像
            history: 对话历史
            temperature: 温度参数
            top_p: top-p采样参数
            max_length: 最大生成长度
            num_beams: beam数
        """
        if not text.strip():
            raise ValueError("输入文本不能为空")

        try:
            start_time = time.time()
            history = history or []

            # 处理输入
            input_text = " ".join(history + [text])
            input_ids = torch.tensor(tokenizer.encode(input_text).ids).unsqueeze(0).to(self.device)

            # 处理图像（如果有）
            image_tensor = self.process_image(image) if image is not None else None

            # 获取最大长度
            if max_length is None:
                max_length = self.config.get('max_length', 100)

            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=self.device == 'cuda'):
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    early_stopping=True if num_beams > 1 else False,
                    do_sample=True,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=2
                )

                response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
                result = response[len(input_text):].strip()

                logger.info(f"生成完成，耗时: {time.time() - start_time:.2f}秒")
                return result

        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            raise RuntimeError(f"生成失败：{str(e)}")

    def batch_predict(self, tokenizer: Tokenizer, queries: List[str],
                      images: Optional[List[Union[str, Image.Image, torch.Tensor]]] = None,
                      histories: Optional[List[List[str]]] = None,
                      temperature: float = 1.0,
                      top_p: float = 0.95,
                      max_length: Optional[int] = None) -> List[str]:
        """批量推理

        Args:
            tokenizer: 分词器实例
            queries: 输入文本列表
            images: 可选输入图像列表
            histories: 对话历史列表
            temperature: 温度参数
            top_p: top-p采样参数
            max_length: 最大生成长度

        Returns:
            List[str]: 生成的响应列表

        Raises:
            RuntimeError: 如果生成失败
        """
        try:
            start_time = time.time()
            batch_size = len(queries)
            histories = histories or [[] for _ in range(batch_size)]

            # 确保历史记录与查询数量匹配
            if len(histories) != batch_size:
                raise ValueError(f"历史记录数量({len(histories)})与查询数量({batch_size})不匹配")

            # 处理输入
            input_texts = [" ".join(history + [query]) for history, query in zip(histories, queries)]
            encoded = [tokenizer.encode(text) for text in input_texts]
            max_len = max(len(e.ids) for e in encoded)

            # 填充到相同长度
            input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
            for i, e in enumerate(encoded):
                input_ids[i, :len(e.ids)] = torch.tensor(e.ids)

            input_ids = input_ids.to(self.device)

            # 处理图像（如果有）
            image_tensors = None
            if images:
                if len(images) != batch_size:
                    raise ValueError(f"图像数量({len(images)})与查询数量({batch_size})不匹配")

                image_tensors = torch.cat([self.process_image(img) for img in images], dim=0)

            # 获取最大长度
            if max_length is None:
                max_length = self.config.get('max_length', 100)

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensors,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )

                responses = []
                for i, ids in enumerate(output_ids):
                    response = tokenizer.decode(ids.tolist())
                    result = response.split(queries[i])[-1].strip()
                    responses.append(result)

                logger.info(
                    f"批量生成完成，耗时: {time.time() - start_time:.2f}秒，平均每个查询: {(time.time() - start_time) / batch_size:.2f}秒")
                return responses

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}")
            raise RuntimeError(f"批量生成失败：{str(e)}")

    def clear_cache(self) -> None:
        """清除模型缓存和GPU缓存"""
        self._cache.clear()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("缓存已清除")

    def stream_predict(self, tokenizer: Tokenizer, query: str,
                       image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
                       history: Optional[List[str]] = None,
                       temperature: float = 1.0,
                       top_p: float = 0.95,
                       max_length: Optional[int] = None,
                       num_beams: int = 1):
        """流式生成响应

        Args:
            参数与predict方法相同

        Yields:
            str: 生成的token
        """
        try:
            history = history or []
            input_text = " ".join(history + [query])
            input_ids = torch.tensor(tokenizer.encode(input_text).ids).unsqueeze(0).to(self.device)

            image_tensor = self.process_image(image) if image is not None else None

            if max_length is None:
                max_length = self.config.get('max_length', 100)

            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                # 使用流式生成
                for output in self.model.stream_generate(
                        input_ids=input_ids,
                        images=image_tensor,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        early_stopping=True if num_beams > 1 else False,
                        do_sample=True
                ):
                    yield tokenizer.decode(output[0].tolist())[len(input_text):].strip()

        except Exception as e:
            logger.error(f"流式生成失败: {str(e)}")
            raise RuntimeError(f"流式生成失败：{str(e)}")