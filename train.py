"""MegaLLM 模型训练脚本

该脚本实现了完整的模型训练流程，包括:
- 数据加载与预处理
- 模型训练与验证
- 自动批次大小调整
- 混合精度训练
- 模型保存与恢复

使用示例:
    python train.py --config config.json

主要类:
    Trainer: 实现核心训练逻辑的类
"""

import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import os
from contextlib import nullcontext
from torchvision import transforms
from PIL import Image

from bpe.tokenizer import BpeTokenizer
from model.architecture import MegaLLM
from utils.config import load_config
from utils.dataloader import LLMDataset

class Trainer:
    """MegaLLM 模型训练器
    
    实现完整的训练流程，包括数据加载、模型训练、验证和保存。
    
    Args:
        config_path (str): 配置文件路径
    """
    def __init__(self, config_path):
        """初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置文件
        self.config = load_config(config_path)

        # 设置设备为GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化模型并移动到GPU
        self.model = MegaLLM(self.config.model).to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        # 初始化优化器
        self.optim = AdamW(self.model.parameters(), lr=self.config.training.lr)

        # 更新GradScaler初始化方式
        self.scaler = GradScaler(init_scale=2.**16) if self.device.type == 'cuda' else None
        
        # 更新autocast使用方式
        self.autocast = torch.amp.autocast('cuda', dtype=torch.float16) if self.device.type == 'cuda' else nullcontext()
        
        # 初始化自动批次大小调整器
        self.auto_batch = self.AutoBatchAdjuster(
            min_batch=4,  # 减小最小批次大小
            max_batch=self.config.training.batch_size
        )
        
        # 设置梯度累积步数
        self.accumulation_steps = self.config.training.accumulation_steps if hasattr(self.config.training, 'accumulation_steps') else 1
        
        # Enable cudnn auto-tuner for convolution speedup if possible
        if self.device.type == 'cuda':
            cudnn.benchmark = True
            print("CUDNN benchmark enabled")
        else:
            print("Running on CPU - performance may be limited")

        # 设置tokenizer路径
        self.tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

        # 初始化tokenizer
        self.tokenizer = BpeTokenizer()  # 确保使用self.tokenizer而不是局部变量tokenizer
        
        # 检查tokenizer.json文件是否存在
        if os.path.exists(self.tokenizer_path):
            print("Loading pre-trained tokenizer from tokenizer.json...")
            try:
                self.tokenizer.load(self.tokenizer_path)  # 使用self.tokenizer
                print(f"Tokenizer loaded successfully with vocab size: {len(self.tokenizer)}")
                
                # 测试tokenizer功能
                test_text = "Hello world"
                encoded = self.tokenizer.encode(test_text)  # 使用self.tokenizer
                decoded = self.tokenizer.decode(encoded)  # 使用self.tokenizer
                print(f"Tokenizer test - Original: '{test_text}', Decoded: '{decoded}'")
                
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                print("Training new tokenizer instead...")
        else:
            print("Tokenizer not found. Will train new tokenizer...")

        # 添加多模态支持
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def load_data(self):
        """加载并预处理训练数据
        
        包括:
        - 加载JSON格式的训练数据
        - 训练或加载BPE分词器
        - 创建DataLoader
        
        Raises:
            ValueError: 如果数据为空或格式不正确
        """
        # 加载训练数据并进行BPE分词
        try:
            with open(self.config.training.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 数据校验和处理多模态数据
            processed_data = []
            for item in data:
                sample = {
                    'prompt': item['prompt'],
                    'completion': item['completion']
                }
                if 'image_path' in item:
                    try:
                        image = Image.open(item['image_path'])
                        sample['image'] = self.image_processor(image)
                    except Exception as e:
                        print(f"加载图像 {item['image_path']} 失败: {e}")
                        continue
                processed_data.append(sample)
            
            # 数据校验
            if not data:
                raise ValueError("训练数据为空，请检查数据文件路径和内容")
            print(f"成功加载 {len(data)} 条训练样本")

            # 将训练数据转化为字符串并加入分词器
            training_text = ' '.join([item['prompt'] + item['completion'] for item in data])
            # 文本长度校验
            if len(training_text) < 100:
                raise ValueError("训练文本过短，至少需要100个字符")
            
            # 确保使用self.tokenizer而不是局部变量tokenizer
            tokenizer = self.tokenizer

            # 如果tokenizer需要训练
            if not os.path.exists(self.tokenizer_path):
                print(f"开始训练tokenizer，目标词汇量：{self.config.model.vocab_size}")
                self.tokenizer.train_from_iterator([training_text], vocab_size=self.config.model.vocab_size, min_freq=2)
                print(f"Tokenizer训练完成，词汇量: {len(self.tokenizer)}")
                
                # 保存新训练的tokenizer
                self.tokenizer.save(self.tokenizer_path)
                print(f"Tokenizer已保存到 {self.tokenizer_path}")

            # 创建自定义数据集并初始化DataLoader
            self.dataset = LLMDataset(self.tokenizer, self.config.training.data_path)  # 这里已正确使用self.tokenizer
            # 数据集校验
            if len(self.dataset) == 0:
                raise ValueError("数据集为空，请检查数据预处理逻辑")
            
            # 输出第一个样本的部分信息，避免打印过多数据
            first_sample = self.dataset[0]
            print(f"数据集包含 {len(self.dataset)} 个样本")
            print(f"首样本输入长度: {len(first_sample['input_ids'])}, 目标长度: {len(first_sample['target_ids'])}")

            self.loader = DataLoader(
                self.dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                pin_memory=True if self.device.type == 'cuda' else False,
                collate_fn=LLMDataset.collate_fn,
                # 修改num_workers为0避免Windows下的多进程问题
                num_workers=0,  
                drop_last=True,
                persistent_workers=False
            )
            
            print(f"已创建DataLoader，包含 {len(self.loader)} 个批次")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise

    class AutoBatchAdjuster:
        """自动批次大小调整器
        
        根据GPU内存使用情况动态调整批次大小
        
        Attributes:
            min_batch (int): 最小批次大小
            max_batch (int): 最大批次大小
            current_batch (int): 当前批次大小
            oom_count (int): OOM错误计数
        """
        def __init__(self, min_batch=8, max_batch=128):
            self.min_batch = min_batch
            self.max_batch = max_batch
            self.current_batch = min_batch
            self.oom_count = 0
            
        def adjust_batch_size(self, loader):
            """调整批次大小"""
            loader = DataLoader(
                loader.dataset,
                batch_size=self.current_batch,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            return loader
            
        def decrease_batch(self):
            """减小批次大小"""
            self.oom_count += 1
            if self.current_batch > self.min_batch:
                self.current_batch = max(self.min_batch, self.current_batch // 2)
                print(f"减小批次大小至 {self.current_batch}")
                
        def increase_batch(self):
            """增加批次大小"""
            if self.oom_count == 0 and self.current_batch < self.max_batch:
                new_batch = min(self.max_batch, self.current_batch * 2)
                if new_batch != self.current_batch:
                    self.current_batch = new_batch
                    print(f"增加批次大小至 {self.current_batch}")
    
    def train_epoch(self):
        """执行一个训练周期
        
        Returns:
            float: 该周期的平均损失
            
        Raises:
            ValueError: 如果没有处理任何批次
        """
        self.model.train()
        total_loss = 0
        processed_batches = 0
        
        # 动态调整批次大小
        self.loader = self.auto_batch.adjust_batch_size(self.loader)
        
        # 创建进度条
        progress_bar = tqdm(self.loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 确保输入和目标张量的维度正确
                inputs = batch['input_ids'].to(self.device)
                targets = batch['target_ids'].to(self.device)
                
                # 添加注意力掩码
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 多模态数据处理
                if 'images' in batch:
                    images = batch['images'].to(self.device)
                    with self.autocast:
                        outputs = self.model(
                            input_ids=inputs,
                            images=images,
                            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
                        )
                else:
                    with self.autocast:
                        # 检查模型是否接受attention_mask参数
                        if hasattr(self.model, 'forward') and 'attention_mask' in self.model.forward.__code__.co_varnames:
                            outputs = self.model(inputs, attention_mask=attention_mask)
                        else:
                            outputs = self.model(inputs)
                
                # 清除梯度
                self.optim.zero_grad()
                
                # 确保维度匹配 - 修改这里的维度处理
                batch_size, seq_len = inputs.size()
                outputs = outputs.view(-1, self.config.model.vocab_size)
                targets = targets.reshape(-1)
                
                # 打印调试信息
                if batch_idx == 0:
                    print(f"输入形状: {inputs.shape}, 输出形状: {outputs.shape}, 目标形状: {targets.shape}")
                
                # 计算损失
                loss = F.cross_entropy(outputs, targets, ignore_index=-100)

                # 仅在有GPU时使用GradScaler
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                        
                total_loss += loss.item()
                processed_batches += 1
                avg_loss = total_loss / processed_batches
                # 报告当前批次大小
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'batch_size': inputs.size(0)
                })
                
            except torch.cuda.OutOfMemoryError:
                # 自动减小批次大小
                self.auto_batch.decrease_batch()
                torch.cuda.empty_cache()  # 立即清理GPU缓存
                continue
                    
            except Exception as e:
                print(f"训练批次 {batch_idx} 时出错: {e}")
                # 打印更详细的错误信息
                import traceback
                print(traceback.format_exc())
                continue

        # 处理剩余的梯度
        if processed_batches % self.accumulation_steps != 0 and self.scaler is not None:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()

        if processed_batches == 0:
            raise ValueError("未处理任何批次。请调整批次大小或检查数据。")

        return total_loss / processed_batches

    def save_model(self, path):
        """保存模型和tokenizer到指定路径
        
        Args:
            path (str): 模型保存路径
            
        Raises:
            Exception: 如果保存失败
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # Save model and tokenizer with extra safety checks
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, path)
            print(f"模型已保存到 {path}")
            
            # 单独保存tokenizer到同一目录
            tokenizer_path = os.path.join(os.path.dirname(path), "tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            print(f"Tokenizer已保存到 {tokenizer_path}")
            
        except Exception as e:
            print(f"保存模型时出错: {e}")
            raise

    def train(self):
        """执行完整训练流程
        
        包括多个训练周期和定期保存模型
        """
        try:
            # 开始训练
            for epoch in range(self.config.training.epochs):
                print(f"\n开始 Epoch {epoch+1}/{self.config.training.epochs}")
                loss = self.train_epoch()
                print(f"Epoch {epoch+1} 完成，平均损失: {loss:.4f}")

                # 每隔指定epoch保存模型
                if (epoch+1) % self.config.training.save_interval == 0 or epoch == self.config.training.epochs - 1:
                    save_path = f"{self.config.training.save_dir}/model_epoch_{epoch+1}.pth"
                    self.save_model(save_path)
                    
        except KeyboardInterrupt:
            print("\n训练被手动中断")
            print("保存最后一个检查点...")
            self.save_model(f"{self.config.training.save_dir}/model_interrupted.pth")
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            raise
            
        print("训练完成!")

if __name__ == "__main__":
    try:
        trainer = Trainer("config.json")
        trainer.load_data()
        trainer.train()
    except Exception as e:
        print(f"程序运行出错: {e}")