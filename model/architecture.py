"""模型架构实现模块

包含以下主要组件:
1. MultiModalEncoder: 多模态编码器
2. TransformerBlock: Transformer基础块
3. MegaLLM: 主模型架构
"""

import torch
import torch.nn as nn
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
        self.config = config
        self.embed = ParallelEmbedding(config.vocab_size, config.dim)
        self.modal_encoder = MultiModalEncoder(config.dim, config.dim)
        
        # 使用更高效的Flash Attention
        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # 添加LoRA适配器
        self.lora_adapters = nn.ModuleDict({
            'query': nn.Linear(config.dim, config.dim, bias=False),
            'key': nn.Linear(config.dim, config.dim, bias=False),
            'value': nn.Linear(config.dim, config.dim, bias=False)
        })
        
        # 添加专家选择门控
        self.expert_gate = nn.Linear(config.dim, config.num_experts)
        
        # 添加模态类型嵌入
        self.modal_type_embeddings = nn.Embedding(2, config.dim)  # 0: text, 1: image
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.dim, 
                config.num_heads, 
                config.num_experts, 
                config.use_longformer,
                use_rotary=True
            ) for _ in range(config.num_layers)
        ])
        
        self.head = nn.Linear(config.dim, config.vocab_size)
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # 添加LayerNorm
        self.final_norm = RMSNorm(config.dim)

    def forward(self, text_ids, images=None, attention_mask=None):
        # 如果没有提供attention_mask，根据pad_token_id创建
        if attention_mask is None:
            attention_mask = (text_ids != self.config.pad_token_id).long()
        batch_size = text_ids.shape[0]
        
        # 文本嵌入
        x = self.embed(text_ids).type(self.dtype)
        
        # 处理图像（如果有）
        if images is not None:
            image_features = self.modal_encoder(images)
            
            # 添加维度检查和调整
            if image_features.size(-1) != x.size(-1):
                image_features = self.vision_proj(image_features)
            
            # 添加模态类型嵌入
            text_type_embeddings = self.modal_type_embeddings(
                torch.zeros(batch_size, x.size(1), dtype=torch.long, device=x.device)
            )
            image_type_embeddings = self.modal_type_embeddings(
                torch.ones(batch_size, 1, dtype=torch.long, device=x.device)
            )
            
            # 确保维度匹配
            x = x + text_type_embeddings
            image_features = image_features + image_type_embeddings
        
        # 通过transformer层
        for layer in self.layers:
            # 传递注意力掩码
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'forward') and 'attention_mask' in layer.attn.forward.__code__.co_varnames:
                # 修改transformer层处理
                for layer in self.layers:
                    # 添加LoRA适配
                    if hasattr(self, 'lora_adapters'):
                        q = x + self.lora_adapters['query'](x)
                        k = x + self.lora_adapters['key'](x)
                        v = x + self.lora_adapters['value'](x)
                        
                        if self.use_flash_attention:
                            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
                        else:
                            x = layer(x, image_features if images is not None else None)
                        
                def forward(self, x, images=None, attention_mask=None):
                    """前向传播
                    
                    参数:
                        x: 输入token序列 [batch_size, seq_len]
                        images: 可选图像输入 [batch_size, channels, height, width]
                        attention_mask: 注意力掩码
                        
                    返回:
                        模型输出
                    """
                    # 确保输入序列长度能被注意力头数整除
                    seq_len = x.size(1)
                    num_heads = self.config.model.num_heads
                    if seq_len % num_heads != 0:
                        padding_size = num_heads - (seq_len % num_heads)
                        x = F.pad(x, (0, padding_size), "constant", 0)
                        
                    # 处理多模态输入
                    image_features = None
                    if images is not None and self.config.model.multimodal:
                        image_features = self.image_encoder(images)
                        
                    # 通过各层处理
                    for layer in self.layers:
                        # 传递注意力掩码
                        if hasattr(layer, 'attn') and hasattr(layer.attn, 'forward') and 'attention_mask' in layer.attn.forward.__code__.co_varnames:
                            x = layer(x, image_features if images is not None else None, attention_mask=attention_mask)
                        else:
                            x = layer(x, image_features if images is not None else None)
                    
                    # 移除可能的填充
                    if seq_len != x.size(1):
                        x = x[:, :seq_len, :]
                        
                    x = self.final_norm(x)
                    return self.head(x.type(torch.float32))
            else:
                x = layer(x, image_features if images is not None else None)
        
        x = self.final_norm(x)
        return self.head(x.type(torch.float32))

    def generate(self, input_ids, images=None, max_length=100, temperature=1.0, top_p=0.95):
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            generated = input_ids
            
            for _ in range(max_length):
                outputs = self(generated[:, -self.config.max_seq_len:], images)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # top-p采样
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for idx in range(batch_size):
                    next_token_logits[idx, sorted_indices[idx][sorted_indices_to_remove[idx]]] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == self.config.eos_token_id).any():
                    break
                    
            return generated
    
    def stream_generate(self, input_ids, images=None, **kwargs):
        """流式生成方法
        
        Args:
            input_ids: 输入token ids
            images: 可选图像输入
            **kwargs: 生成参数
            
        Yields:
            torch.Tensor: 生成的token ids
        """
        # 初始化生成状态
        generated = input_ids
        past_key_values = None
        
        for _ in range(kwargs.get('max_length', 100)):
            with torch.no_grad():
                outputs = self(
                    input_ids=generated[:, -1:] if past_key_values is not None else input_ids,
                    images=images,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # 应用温度采样
            next_token_logits = next_token_logits / kwargs.get('temperature', 1.0)
            
            # 应用top-p采样
            if kwargs.get('top_p', 1.0) < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除低概率token
                sorted_indices_to_remove = cumulative_probs > kwargs['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')
                
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=-1)
            yield generated
            
            # 检查停止条件
            if next_token.item() == self.config.eos_token_id:
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
