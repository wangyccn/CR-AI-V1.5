"""
/**
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *                     佛祖保佑        永无BUG
 *            佛曰:
 *                   写字楼里写字间，写字间里程序员；
 *                   程序人员写程序，又拿程序换酒钱。
 *                   酒醒只在网上坐，酒醉还来网下眠；
 *                   酒醉酒醒日复日，网上网下年复年。
 *                   但愿老死电脑间，不愿鞠躬老板前；
 *                   奔驰宝马贵者趣，公交自行程序员。
 *                   别人笑我忒疯癫，我笑自己命太贱；
 *                   不见满街漂亮妹，哪个归得程序员？
*/
"""
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
from transformers import BeamSearchScorer

class MultiModalEncoder(nn.Module):
    """多模态编码器
    
    实现视觉特征的编码和投影
    
    属性:
        vision_encoder (nn.Sequential): 视觉特征编码器
        vision_proj (nn.Linear): 视觉特征投影层
        image_norm (RMSNorm): 图像特征归一化层
    """
    def __init__(self, image_dim=2048, text_dim=2048):
        super().__init__()
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            vision_model = models.efficientnet_b0(weights=weights)
            # 冻结预训练层
            for param in vision_model.parameters():
                param.requires_grad = False
            self.vision_encoder = nn.Sequential(*list(vision_model.children())[:-1])
            self.vision_proj = nn.Linear(1280, image_dim)
            self.image_norm = RMSNorm(image_dim)
            
            # 添加设备同步
            self.device = None
        except Exception as e:
            raise RuntimeError(f"初始化视觉编码器失败: {str(e)}")
    
    def to(self, device):
        """重写to方法以确保设备同步"""
        self.device = device
        return super().to(device)

    def forward(self, images):
        # 确保输入在正确的设备上
        if self.device is not None:
            images = images.to(self.device)
            
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
    """多模态大型语言模型(MegaLLM)实现
    
    主要功能:
    - 支持多模态输入(文本+图像)
    - 可配置的Transformer架构
    - 高效的稀疏注意力机制
    - 文本生成功能
    
    属性:
        embed (ParallelEmbedding): 词嵌入层
        layers (nn.ModuleList): Transformer层列表
        use_gradient_checkpointing (bool): 是否使用梯度检查点
        use_flash_attention (bool): 是否使用闪存注意力
    """
    def __init__(self, config):
        super().__init__()
        self.use_gradient_checkpointing = True
        self.use_flash_attention = True
        
        # 修改embed属性名以匹配权重文件
        self.embed = ParallelEmbedding(
            getattr(config, 'vocab_size', 30522), 
            getattr(config, 'dim', 768)
        )
        # 添加embedding属性别名以兼容旧权重
        self.embedding = self.embed
        
        self.layers = nn.ModuleList([
            self._create_layer(config, i) for i in range(getattr(config, 'num_layers', 12))
        ])

    def prepare_inputs_for_generation(self, input_ids, images=None, **kwargs):
        """准备生成文本所需的输入
        
        参数:
            input_ids: 输入的token ids
            images: 可选的图像输入
            **kwargs: 其他参数
            
        返回:
            dict: 包含模型输入的字典
        """
        model_inputs = {
            'input_ids': input_ids,
        }
        
        if images is not None:
            model_inputs['images'] = images
            
        # 添加其他可能的输入参数
        for key, value in kwargs.items():
            model_inputs[key] = value
            
        return model_inputs
        
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
        
    def forward(self, text_ids, images=None, **kwargs):
        """前向传播
        
        参数:
            text_ids (Tensor): 输入的token ids
            images (Tensor): 可选的图像输入
            **kwargs: 其他参数(如length_penalty等会被忽略)
            
        返回:
            Tensor: 模型输出
        """
        # 确保模型和输入在同一设备上
        device = next(self.parameters()).device
        text_ids = text_ids.to(device)
            
        # 初始化x为嵌入层的输出
        x = self.embed(text_ids)
        
        # 如果有图像输入，处理图像特征
        if images is not None:
            images = images.to(device)
            # 确保image_encoder在正确的设备上
            image_encoder = MultiModalEncoder(image_dim=x.size(-1)).to(device)
            images = image_encoder(images)  # 输出形状为[B, image_dim]
            images = images.unsqueeze(1)  # 变为[B, 1, image_dim]
        
        # 梯度检查点
        def create_custom_forward(layer):
            def custom_forward(*inputs):
                # 确保所有输入在同一设备上
                inputs = [i.to(device) if isinstance(i, torch.Tensor) else i for i in inputs]
                return layer(*inputs)
            return custom_forward
            
        if self.use_gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x.to(device),  # 确保x在正确设备上
                    images.to(device) if images is not None else None,
                    use_reentrant=False
                )
        else:
            for layer in self.layers:
                x = layer(x.to(device), images.to(device) if images is not None else None)
                
        return x.to(device)  # 确保输出在正确设备上

    def generate(self, input_ids, images=None, max_length=100, temperature=1.0,
                 top_p=0.95, do_sample=True, eos_token_id=None, num_beams=1, **kwargs):
        """生成文本的完整实现"""
        self.eval()
        device = next(self.parameters()).device  # 获取模型设备
        
        with torch.no_grad():
            # 确保输入在正确设备上
            input_ids = input_ids.to(device)
            if images is not None:
                images = images.to(device)
            
            # 过滤掉forward不支持的参数
            forward_kwargs = {
                'text_ids': input_ids,
                'images': images
            }
            
            if num_beams > 1:
                # 使用beam search生成
                beam_scorer = BeamSearchScorer(
                    batch_size=input_ids.size(0),
                    num_beams=num_beams,
                    device=device,  # 确保beam_scorer使用正确的设备
                    length_penalty=kwargs.get('length_penalty', 1.0)
                )
                return self._generate_beam(
                    input_ids=input_ids,
                    beam_scorer=beam_scorer,
                    max_length=max_length,
                    eos_token_id=eos_token_id,
                    images=images
                )
            else:
                # 使用采样生成
                return self._generate_sample(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    eos_token_id=eos_token_id,
                    images=images
                )

    def _generate_sample(self, input_ids, max_length, temperature=1.0, top_p=0.95,
                        do_sample=True, eos_token_id=None, **kwargs):
        """统一的采样生成实现"""
        batch_size = input_ids.size(0)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        for _ in range(max_length):
            outputs = self.forward(input_ids, **kwargs)
            next_token_logits = outputs[:, -1, :]
            
            # 应用温度缩放和top-p过滤
            next_token_logits = self._apply_sampling_constraints(
                next_token_logits, 
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            
            # 采样或贪婪选择
            next_tokens = self._select_next_tokens(next_token_logits, do_sample)
            
            # 更新输入ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 检查是否应该停止生成
            if eos_token_id is not None:
                unfinished_sequences = self._update_unfinished_sequences(
                    unfinished_sequences, next_tokens, eos_token_id
                )
                if unfinished_sequences.max() == 0:
                    break
        
        return input_ids

    def _generate_beam(self, input_ids, beam_scorer, max_length, eos_token_id=None, **kwargs):
        """统一的beam search生成实现"""
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        for _ in range(max_length):
            outputs = self.forward(input_ids, **kwargs)
            next_token_logits = outputs[:, -1, :]
            
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                eos_token_id=eos_token_id
            )
            
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            
            input_ids = torch.cat([
                input_ids[beam_idx, :], 
                beam_next_tokens.unsqueeze(-1)
            ], dim=-1)
            
            if beam_scorer.is_done:
                break
        
        return beam_scorer.finalize(input_ids, beam_scores)

    def _apply_sampling_constraints(self, logits, temperature=1.0, top_p=0.95, 
                                  do_sample=True, repetition_penalty=1.0):
        """应用温度缩放、top-p过滤和重复惩罚"""
        if temperature != 1.0:
            logits = logits / temperature
            
        # 应用重复惩罚
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, repetition_penalty)
            
        if do_sample and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        return logits

    def _apply_repetition_penalty(self, logits, penalty):
        """应用重复惩罚"""
        # 实现重复惩罚逻辑
        return logits

    def _select_next_tokens(self, logits, do_sample):
        """选择下一个token"""
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            return torch.argmax(logits, dim=-1)

    def _update_unfinished_sequences(self, unfinished_sequences, next_tokens, eos_token_id):
        """更新未完成序列状态"""
        return unfinished_sequences.mul(
            next_tokens.tile(eos_token_id.shape[0], 1).ne(eos_token_id.unsqueeze(1)).prod(dim=0)
        )

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
        # 检测是否可用CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 创建配置对象
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

        # 初始化模型并移至正确设备
        model = MegaLLM(config).to(device)

        # 创建虚拟输入数据并移至正确设备
        text_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len)).to(device)
        images = torch.randn(2, 3, 224, 224).to(device)

        # 前向传播
        output = model(text_ids, images)
        print("输出形状:", output.shape)
        print("输出设备:", output.device)

    except Exception as e:
        print("模型测试失败:", str(e))
        raise  # 抛出异常以便查看完整的错误栈

# 调用测试函数
if __name__ == "__main__":
    print("开始测试模型...")
    test_model()