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
import time

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

    # 将load_data方法移出__init__方法，使其成为Trainer类的成员方法
    def load_data(self):
        """加载并预处理训练数据"""
        try:
            with open(self.config.training.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            # 初始化处理后的数据列表
            processed_data = []
        
            for item in data:
                # 确保输入和目标序列的长度不超过模型的最大长度
                max_length = self.config.model.max_length if hasattr(self.config.model, 'max_length') else 2048
                
                input_tokens = self.tokenizer.encode(item['prompt'])[:max_length]
                target_tokens = self.tokenizer.encode(item['completion'])[:max_length]
                
                sample = {
                    'input_ids': torch.tensor(input_tokens, dtype=torch.long),
                    'target_ids': torch.tensor(target_tokens, dtype=torch.long),
                    'prompt': item['prompt'],
                    'completion': item['completion']
                }
            
                # 处理多模态数据
                if 'image_path' in item:
                    try:
                        image = Image.open(item['image_path'])
                        sample['images'] = self.image_processor(image)
                    except Exception as e:
                        print(f"加载图像 {item['image_path']} 失败: {e}")
                        continue
                    
                processed_data.append(sample)
        
            # 数据校验
            if not processed_data:
                raise ValueError("训练数据为空，请检查数据文件路径和内容")
            print(f"成功加载 {len(processed_data)} 条训练样本")

                # 将训练数据转化为字符串并加入分词器
            training_text = ' '.join([item['prompt'] + item['completion'] for item in processed_data])
                # 文本长度校验
            if len(training_text) < 100:
                raise ValueError("训练文本过短，至少需要100个字符")
            
                # 如果tokenizer需要训练
            if not os.path.exists(self.tokenizer_path):
                print(f"开始训练tokenizer，目标词汇量：{self.config.model.vocab_size}")
                self.tokenizer.train_from_iterator([training_text], vocab_size=self.config.model.vocab_size, min_freq=2)
                print(f"Tokenizer训练完成，词汇量: {len(self.tokenizer)}")
                
                # 保存新训练的tokenizer
                self.tokenizer.save(self.tokenizer_path)
                print(f"Tokenizer已保存到 {self.tokenizer_path}")

            # 创建自定义数据集并初始化DataLoader
            # 将处理后的数据保存为临时JSON文件
            # 修改processed_data中的数据结构，将Tensor转换为列表
            json_safe_data = []
            for item in processed_data:
                json_item = {
                    'input_ids': item['input_ids'].tolist(),  # 转换为列表
                    'target_ids': item['target_ids'].tolist(),  # 转换为列表
                    'prompt': item['prompt'],
                    'completion': item['completion']
                }
                if 'images' in item:
                    json_item['images'] = item['images'].tolist()  # 如果有图像数据，也转换为列表
                json_safe_data.append(json_item)

            # 保存JSON安全的数据
            temp_data_path = os.path.join(os.path.dirname(__file__), "temp_data.json")
            with open(temp_data_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_data, f, ensure_ascii=False)
                
            self.dataset = LLMDataset(self.tokenizer, temp_data_path)
                # 数据集校验
            if len(self.dataset) == 0:
                    raise ValueError("数据集为空，请检查数据预处理逻辑")
            
                # 输出第一个样本的部分信息
            first_sample = self.dataset[0]
            print(f"数据集包含 {len(self.dataset)} 个样本")
            print(f"首样本输入长度: {len(first_sample['input_ids'])}, 目标长度: {len(first_sample['target_ids'])}")

                # 计算实际批次大小
            effective_batch_size = max(1, min(
                self.config.training.batch_size,
                len(self.dataset)
            ))
            
            self.loader = DataLoader(
                self.dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                pin_memory=True if self.device.type == 'cuda' else False,
                collate_fn=LLMDataset.collate_fn,
                num_workers=0,
                drop_last=True,
                persistent_workers=False
            )
            
            print(f"已创建DataLoader，批次大小: {effective_batch_size}, 包含 {len(self.loader)} 个批次")
            
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
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        
        for step, batch in enumerate(progress):
            try:
                # 确保数据正确加载到设备
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target_ids'].to(self.device)
                images = batch.get('images', None)
                if images is not None:
                    images = images.to(self.device)
                    
                # 混合精度训练
                with self.autocast:
                    outputs = self.model(input_ids, images=images)
                    # 确保输出和目标的形状匹配
                    outputs = outputs[:, :targets.size(1), :]  # 裁剪输出以匹配目标长度
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                    # 修改view为reshape
                    loss = F.cross_entropy(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1),
                        ignore_index=pad_token_id
                    )
                    loss = loss / self.accumulation_steps
                
                # 梯度累积
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (step + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.scaler:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # 参数更新
                    if self.scaler:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps  # 修正损失累加
                progress.set_postfix({'loss': f"{total_loss/(step+1):.4f}"})

            except Exception as e:
                print(f"处理批次 {step} 时出错: {e}")
                import traceback
                traceback.print_exc()  # 添加详细错误信息
                continue
                
        return total_loss / len(self.loader)

    def save_model(self, path):
        """保存模型到指定路径"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 使用更可靠的保存方式
            checkpoint = {
                'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'optimizer_state_dict': {k: v.cpu() for k, v in self.optim.state_dict().items()},
                'config': self.config
            }
            
            # 使用临时文件保存
            temp_path = path + '.tmp'
            # 使用更安全的保存方式
            torch.save(checkpoint, temp_path, 
                     _use_new_zipfile_serialization=True,
                     pickle_protocol=4)  # 使用更兼容的pickle协议
            
            # 确保文件完全写入磁盘
            if os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    pass  # 确保文件已关闭
                
            # 重命名确保原子性
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
            
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            raise

    def train(self):
        """执行完整训练流程"""
        try:
            # 开始训练
            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch + 1  # 添加当前epoch记录
                print(f"\n开始 Epoch {self.current_epoch}/{self.config.training.epochs}")
                loss = self.train_epoch(self.current_epoch)
                print(f"Epoch {self.current_epoch} 完成，平均损失: {loss:.4f}")
                
                # 每隔指定epoch保存模型
                if (self.current_epoch) % self.config.training.save_interval == 0 or epoch == self.config.training.epochs - 1:
                    save_path = f"{self.config.training.save_dir}/model_epoch_{self.current_epoch}.pth"
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
    import argparse
    
    parser = argparse.ArgumentParser(description="MegaLLM 模型训练脚本")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    try:
        if args.debug:
            print("调试模式已启用，将输出详细错误信息")
            
        trainer = Trainer(args.config)
        trainer.load_data()
        trainer.train()
    except Exception as e:
        print(f"程序运行出错: {e}")
