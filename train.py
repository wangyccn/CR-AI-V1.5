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

# 优化导入部分
import json
import os
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from bpe.tokenizer import BpeTokenizer
from model.architecture import MegaLLM
from utils.config import load_config
from utils.dataloader import LLMDataset

# 加载配置
config = load_config("config.json")  # 确保路径正确

# 初始化模型
model = MegaLLM(config.model)


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
        # 初始化所有实例属性
        self.config = None
        self.device = None
        self.model = None
        self.optim = None
        self.scaler = None
        self.autocast = None
        self.auto_batch = None
        self.accumulation_steps = None
        self.tokenizer_path = None
        self.tokenizer = None
        self.image_processor = None
        self.dataset = None
        self.loader = None
        self.current_epoch = None

        # 加载配置文件
        self.config = load_config(config_path)

        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = MegaLLM(self.config.model).to(self.device)
        print(
            f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        # 优化器
        self.optim = AdamW(self.model.parameters(), lr=self.config.training.lr)

        # 混合精度训练设置
        self.scaler = GradScaler(
            init_scale=2.**16) if self.device.type == 'cuda' else None
        self.autocast = torch.amp.autocast(
            'cuda', dtype=torch.float16) if self.device.type == 'cuda' else nullcontext()

        # 自动批次大小调整
        self.auto_batch = self.AutoBatchAdjuster(
            min_batch=4,
            max_batch=self.config.training.batch_size
        )

        # 梯度累积步数
        self.accumulation_steps = getattr(
            self.config.training, 'accumulation_steps', 1)

        # CUDNN设置
        if self.device.type == 'cuda':
            cudnn.benchmark = True
            print("CUDNN benchmark enabled")
        else:
            print("Running on CPU - performance may be limited")

        # Tokenizer设置
        self.tokenizer_path = os.path.join(
            os.path.dirname(__file__), "tokenizer.json")
        self.tokenizer = BpeTokenizer()

        # 多模态支持
        self.image_processor = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def load_data(self):
        """加载并预处理训练数据"""
        try:
            with open(self.config.training.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            processed_data = []
            for item in data:
                max_length = self.config.model.max_length
                input_tokens = self.tokenizer.encode(item['prompt'])[
                    :max_length]
                target_tokens = self.tokenizer.encode(
                    item['completion'])[:max_length]

                if isinstance(input_tokens[0], str):
                    input_tokens = [int(t) for t in input_tokens]
                if isinstance(target_tokens[0], str):
                    target_tokens = [int(t) for t in target_tokens]

                sample = {
                    'input_ids': torch.tensor(
                        input_tokens,
                        dtype=torch.long),
                    'target_ids': torch.tensor(
                        target_tokens,
                        dtype=torch.long),
                    'prompt': item['prompt'],
                    'completion': item['completion']}

                if 'image_path' in item:
                    try:
                        image = Image.open(item['image_path'])
                        sample['images'] = self.image_processor(image)
                    except IOError as e:
                        print(f"加载图像 {item['image_path']} 失败: {e}")
                        continue

                processed_data.append(sample)

            if not processed_data:
                raise ValueError("训练数据为空，请检查数据文件路径和内容")
            print(f"成功加载 {len(processed_data)} 条训练样本")

            # 将训练数据转化为字符串并加入分词器
            training_text = ' '.join(
                [item['prompt'] + item['completion'] for item in processed_data])
            # 文本长度校验
            if len(training_text) < 100:
                raise ValueError("训练文本过短，至少需要100个字符")

            # 如果tokenizer需要训练
            if not os.path.exists(self.tokenizer_path):
                print(f"开始训练tokenizer，目标词汇量：{self.config.model.vocab_size}")
                self.tokenizer.train_from_iterator(
                    [training_text],
                    vocab_size=self.config.model.vocab_size,
                    min_freq=2
                )
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
                    # 如果有图像数据，也转换为列表
                    json_item['images'] = item['images'].tolist()
                json_safe_data.append(json_item)

            # 保存JSON安全的数据
            temp_data_path = os.path.join(
                os.path.dirname(__file__), "temp_data.json")
            with open(temp_data_path, 'w', encoding='utf-8') as f:
                json.dump(json_safe_data, f, ensure_ascii=False)

            self.dataset = LLMDataset(self.tokenizer, temp_data_path)
            # 数据集校验
            if len(self.dataset) == 0:
                raise ValueError("数据集为空，请检查数据预处理逻辑")

            # 输出第一个样本的部分信息
            first_sample = self.dataset[0]
            print(f"数据集包含 {len(self.dataset)} 个样本")
            print(
                f"首样本输入长度: {len(first_sample['input_ids'])}, 目标长度: {len(first_sample['target_ids'])}")

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

            print(
                f"已创建DataLoader，批次大小: {effective_batch_size}, 包含 {len(self.loader)} 个批次")

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
                self.current_batch = max(
                    self.min_batch, self.current_batch // 2)
                print(f"减小批次大小至 {self.current_batch}")

        def increase_batch(self):
            """增加批次大小"""
            if self.oom_count == 0 and self.current_batch < self.max_batch:
                new_batch = min(self.max_batch, self.current_batch * 2)
                if new_batch != self.current_batch:
                    self.current_batch = new_batch
                    print(f"增加批次大小至 {self.current_batch}")

    def train_epoch(self, epoch):
        """执行单个训练轮次

        Args:
            epoch (int): 当前训练轮次

        Returns:
            float: 该轮次的平均损失值
        """
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
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)

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

            # 简化保存逻辑，直接保存模型状态
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'config': self.config
            }, path)

            print(f"模型已成功保存到 {path}")

        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            raise

    def train(self):
        """执行完整训练流程"""
        try:
            # 使用getattr提供默认值
            total_epochs = int(getattr(self.config.training, 'epochs', 100))
            save_interval = int(
                getattr(
                    self.config.training,
                    'save_interval',
                    10))

            print(f"\n总训练轮次: {total_epochs}")

            # 开始训练
            for epoch in range(total_epochs):
                self.current_epoch = epoch + 1
                print(f"\n开始 Epoch {self.current_epoch}/{total_epochs}")
                loss = self.train_epoch(self.current_epoch)
                print(f"Epoch {self.current_epoch} 完成，平均损失: {loss:.4f}")

                # 每隔指定epoch保存模型
                if (self.current_epoch) % save_interval == 0:
                    save_path = f"{self.config.training.save_dir}/model_epoch_{self.current_epoch}.pth"
                    self.save_model(save_path)

            # 训练结束时保存最终模型到运行目录
            final_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.config.model.model_path
            )
            self.save_model(final_path)
            print(f"最终模型已保存到运行目录: {os.path.abspath(final_path)}")

        except KeyboardInterrupt:
            print("\n训练被手动中断")
            save_path = f"{self.config.training.save_dir}/model_interrupted.pth"
            self.save_model(save_path)

        except Exception as e:
            print(f"训练过程中出错: {e}")
            raise

        print("训练完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MegaLLM 模型训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="配置文件路径")
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
