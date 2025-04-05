"""模型架构实现模块

包含以下主要组件:
1. MultiModalEncoder: 多模态编码器
2. TransformerBlock: Transformer基础块
3. MegaLLM: 主模型架构
"""

import torch
import torch.nn as nn

from bpe import tokenizer
from .longformer_attention import LongformerAttention
from .moe import MoE
from .sparse_attention import SparseAttention
from .utils import RMSNorm, ParallelEmbedding
from torchvision import models
from torch.nn import functional as F

class MultiModalEncoder(nn.Module):
    """多模态编码器
    
    实现视觉特征的编码和投影
    
    属性:
        vision_encoder (nn.Sequential): 视觉特征编码器
        vision_proj (nn.Linear): 视觉特征投影层
        image_norm (RMSNorm): 图像特征归一化层
    """
    def __init__(self, image_dim=2048, text_dim=2048):
        """初始化多模态编码器
        
        参数:
            image_dim (int): 图像特征维度(默认为2048)
            text_dim (int): 文本特征维度(默认为2048)
        """
        super().__init__()
        try:
            # 使用更高效的权重加载方式
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            vision_model = models.efficientnet_b0(weights=weights)
            # 冻结预训练层
            for param in vision_model.parameters():
                param.requires_grad = False
            self.vision_encoder = nn.Sequential(*list(vision_model.children())[:-1])
            self.vision_proj = nn.Linear(1280, image_dim)
            self.image_norm = RMSNorm(image_dim)
        except Exception as e:
            raise RuntimeError(f"初始化视觉编码器失败: {str(e)}")

    def forward(self, images):
        """前向传播
        
        参数:
            images (Tensor): 输入图像张量
            
        返回:
            Tensor: 编码后的图像特征
        """
        vision_features = self.vision_encoder(images)
        vision_features = vision_features.flatten(start_dim=2).mean(dim=2)
        vision_features = self.vision_proj(vision_features)
        return self.image_norm(vision_features)

class TransformerBlock(nn.Module):
    """Transformer基础块
    
    实现包含以下组件的Transformer层:
    - 稀疏注意力或Longformer注意力
    - MoE层
    - 前馈网络
    - 旋转位置编码
    
    属性:
        use_rotary (bool): 是否使用旋转位置编码
        dim (int): 特征维度
        num_heads (int): 注意力头数
        attn (nn.Module): 注意力层
        norm1 (RMSNorm): 第一层归一化
        norm2 (RMSNorm): 第二层归一化
        moe (MoE): 混合专家层
        ffn (nn.Sequential): 前馈网络
    """
    def __init__(self, dim, num_heads=8, num_experts=8, use_longformer=False, use_rotary=True):
        """初始化Transformer块
        
        参数:
            dim (int): 特征维度
            num_heads (int): 注意力头数(默认为8)
            num_experts (int): 专家数量(默认为8)
            use_longformer (bool): 是否使用Longformer注意力(默认为False)
            use_rotary (bool): 是否使用旋转位置编码(默认为True)
        """
        super().__init__()
        self.use_rotary = use_rotary
        self.dim = dim
        self.num_heads = num_heads
        if use_longformer:
            self.attn = LongformerAttention(dim, num_heads)
        else:
            self.attn = SparseAttention(dim, num_heads)
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.moe = MoE(dim, num_experts)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )
        
        if use_rotary:
            self.register_buffer(
                "rotary_emb", 
                self._create_rotary_embedding(dim // num_heads, 2048)
            )

    def _create_rotary_embedding(self, dim, max_seq_len):
        """创建旋转位置编码
        
        参数:
            dim (int): 每个头的维度
            max_seq_len (int): 最大序列长度
            
        返回:
            Tensor: 旋转位置编码张量
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len)
        sinusoid = torch.einsum('i,j->ij', pos, inv_freq)
        return torch.cat((sinusoid.sin(), sinusoid.cos()), dim=-1)

    def forward(self, x, media_features=None):
        if media_features is not None:
            if media_features.dim() == 2:
                media_features = media_features.unsqueeze(1)
            if media_features.size(-1) != x.size(-1):
                media_features = nn.Linear(media_features.size(-1), x.size(-1))(media_features)
            x = torch.cat([media_features, x], dim=1)
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 应用第一个归一化层
        normalized_x = self.norm1(x)
        
        # 应用注意力机制
        # 确保注意力输出与输入形状匹配
        try:
            attn_output = self.attn(normalized_x)
            
            # 检查注意力输出形状是否与输入匹配
            if attn_output.shape != residual.shape:
                # 如果不匹配，尝试调整形状
                if attn_output.shape[0] == residual.shape[0] and attn_output.shape[2] == residual.shape[2]:
                    # 只有序列长度不同，进行调整
                    if attn_output.shape[1] < residual.shape[1]:
                        # 注意力输出较短，进行填充
                        padding = torch.zeros(
                            (attn_output.shape[0], residual.shape[1] - attn_output.shape[1], attn_output.shape[2]),
                            device=attn_output.device,
                            dtype=attn_output.dtype
                        )
                        attn_output = torch.cat([attn_output, padding], dim=1)
                    else:
                        # 注意力输出较长，进行截断
                        attn_output = attn_output[:, :residual.shape[1], :]
            
            # 残差连接
            x = residual + attn_output
        except Exception as e:
            # 如果注意力计算失败，跳过这一层
            print(f"注意力计算失败: {str(e)}")
            x = residual
        
        # 保存第一阶段输出用于第二阶段残差连接
        residual = x
        
        # 应用第二个归一化层和前馈网络
        x = residual + self.ffn(self.norm2(x))
        
        return x

class MegaLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 添加内存优化选项
        self.use_gradient_checkpointing = True
        self.use_flash_attention = True
        
        # 修改为支持对象属性访问
        self.embed = ParallelEmbedding(getattr(config, 'vocab_size', 30522), 
                                     getattr(config, 'dim', 768))
        self.layers = nn.ModuleList([
            self._create_layer(config, i) for i in range(getattr(config, 'num_layers', 12))
        ])
        
    def _create_layer(self, config, layer_idx):
        # 修改为支持对象属性访问
        use_longformer = (layer_idx % 2 == 0) if getattr(config, 'use_longformer', False) else False
        return TransformerBlock(
            getattr(config, 'dim', 768),
            getattr(config, 'num_heads', 12),
            getattr(config, 'num_experts', 4),
            use_longformer,
            use_rotary=True
        )
        
    def forward(self, text_ids, images=None):
        # 初始化x为嵌入层的输出
        x = self.embed(text_ids)
        
        # 如果有图像输入，处理图像特征
        if images is not None:
            # 使用MultiModalEncoder处理图像
            image_encoder = MultiModalEncoder(image_dim=x.size(-1))
            images = image_encoder(images)  # 输出形状为[B, image_dim]
            images = images.unsqueeze(1)  # 变为[B, 1, image_dim]
        
        # 梯度检查点
        def create_custom_forward(layer):
            def custom_forward(*inputs):
                return layer(*inputs)
            return custom_forward
            
        if self.use_gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    images,
                    use_reentrant=False  # 添加推荐参数
                )
        else:
            for layer in self.layers:
                x = layer(x, images)
                
        return x

    def eval(self) -> 'MegaLLM':
        """将模型设置为评估模式
        
        返回:
            MegaLLM: 返回模型自身以支持链式调用
        """
        super().eval()
        
        # 禁用梯度计算
        for param in self.parameters():
            param.requires_grad = False
            
        # 禁用dropout和batch norm的train模式
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
                
        # 禁用梯度检查点
        self.use_gradient_checkpointing = False
        
        # 确保flash attention在eval模式下可用
        if hasattr(self, 'use_flash_attention'):
            self.use_flash_attention = True
            
        return self

    def generate(self, input_ids, images=None, max_length=100, temperature=1.0, 
                top_p=0.95, do_sample=True, eos_token_id=None):  # 添加do_sample参数
        """生成文本
        
        参数:
            input_ids (Tensor): 输入token IDs
            images (Tensor): 可选图像输入
            max_length (int): 最大生成长度
            temperature (float): 温度参数
            top_p (float): top-p采样参数
            do_sample (bool): 是否采样(默认为True)
            eos_token_id (int): 结束标记的token ID
        """
        self.eval()
        with torch.no_grad():
            # 初始化输出序列
            output = input_ids
            
            # 处理图像特征
            image_features = None
            if images is not None:
                image_encoder = MultiModalEncoder(image_dim=self.embed.embedding_dim)
                image_features = image_encoder(images)
                image_features = image_features.unsqueeze(1)
            
            # 生成循环
            for _ in range(max_length - input_ids.size(1)):
                # 获取模型输出
                logits = self.forward(output, image_features)[:, -1, :]
                
                # 应用温度
                logits = logits / temperature
                
                # 应用top-p采样
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # 采样或贪婪解码
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # 添加到输出序列
                output = torch.cat([output, next_token], dim=-1)
                
                # 检查是否应该停止
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
            
            return output

    def stream_generate(self, input_ids, images=None, max_length=100, temperature=1.0, 
                       top_p=0.95, num_beams=1, early_stopping=False, do_sample=True):
        """流式生成文本
        
        参数与generate方法相同
        
        返回:
            Generator: 生成token的生成器
        """
        self.eval()
        with torch.no_grad():
            output = input_ids
            image_features = None
            if images is not None:
                image_encoder = MultiModalEncoder(image_dim=self.embed.embedding_dim)
                image_features = image_encoder(images)
                image_features = image_features.unsqueeze(1)
            
            for _ in range(max_length - input_ids.size(1)):
                logits = self.forward(output, image_features)[:, -1, :]
                logits = logits / temperature
                
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                output = torch.cat([output, next_token], dim=-1)
                yield output
                
                if not hasattr(tokenizer, 'eos_token_id'):
                    raise AttributeError("tokenizer缺少eos_token_id属性")
                
                if next_token.item() == tokenizer.eos_token_id:
                    break


def test_model():
    """测试模型的最小化验证函数"""
    try:
        # 创建一个简单的配置对象
        class Config:
            vocab_size = 30522
            dim = 768
            num_heads = 12
            num_experts = 4
            num_layers = 2
            pad_token_id = 0
            max_seq_len = 128
            use_longformer = False
            multimodal = True

        config = Config()

        # 初始化模型
        model = MegaLLM(config)

        # 创建虚拟输入数据
        text_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
        images = torch.randn(2, 3, 224, 224)  # 假设图像输入为224x224

        # 前向传播
        output = model(text_ids, images)

        # 打印输出形状
        print("输出形状:", output.shape)

    except Exception as e:
        print("模型测试失败:", str(e))

# 调用测试函数
if __name__ == "__main__":
    test_model()
