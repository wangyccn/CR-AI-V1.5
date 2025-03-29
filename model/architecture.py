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
                media_features = self.norm1(media_features)
            x = torch.cat([media_features, x], dim=1)
        
        if self.use_rotary:
            seq_len = x.shape[1]
            rotary = self.rotary_emb[:seq_len]
            # 修正维度处理
            rotary = rotary.view(1, seq_len, 1, -1)  # [1, seq_len, 1, dim//num_heads]
            rotary = rotary.repeat(1, 1, self.num_heads, 1)  # [1, seq_len, num_heads, dim//num_heads]
            rotary = rotary.permute(0, 2, 1, 3)  # [1, num_heads, seq_len, dim//num_heads]
            
            # 调整x的维度以匹配旋转位置编码
            x = x.view(x.size(0), seq_len, self.num_heads, -1)  # [batch, seq_len, num_heads, dim//num_heads]
            x = x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, dim//num_heads]
            x = x * rotary
            x = x.permute(0, 2, 1, 3)  # [batch, seq_len, num_heads, dim//num_heads]
            x = x.contiguous().view(x.size(0), seq_len, -1)  # [batch, seq_len, dim]
            
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
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
            x = x.view(batch_size, -1, self.config.dim)
            image_features = image_features.view(batch_size, -1, self.config.dim)
            
            x = x + text_type_embeddings
            image_features = image_features.unsqueeze(1) + image_type_embeddings
        
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
                        
                x = layer(x, image_features if images is not None else None, attention_mask=attention_mask)
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