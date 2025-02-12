import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import os  # 新增导入 os 模块，用于检查文件是否存在

from bpe.tokenizer import BpeTokenizer
from model.architecture import MegaLLM
from utils.config import load_config
from utils.dataloader import LLMDataset

class Trainer:
    def __init__(self, config_path):
        # 加载配置文件
        self.config = load_config(config_path)

        # 设置设备为GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型并移动到GPU
        self.model = MegaLLM(self.config.model).to(self.device)

        # 初始化优化器
        self.optim = AdamW(self.model.parameters(), lr=self.config.training.lr)

        # 使用GradScaler来支持混合精度训练
        self.scaler = GradScaler()

        # Enable cudnn auto-tuner for convolution speedup if possible
        cudnn.benchmark = True

        # 设置tokenizer路径
        self.tokenizer_path = str(__path__ / "tokenizer.json")

        # 检查tokenizer.json文件是否存在
        if os.path.exists(self.tokenizer_path):
            print("Loading pre-trained tokenizer from tokenizer.json...")
            self.tokenizer = BpeTokenizer.load(self.tokenizer_path)  # 加载现有tokenizer
        else:
            print("Tokenizer not found. Training new tokenizer...")
            self.tokenizer = BpeTokenizer()  # 初始化一个新的tokenizer

    def load_data(self):
        # 加载训练数据并进行BPE分词
        with open(self.config.training.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 将训练数据转化为字符串并加入分词器
        training_text = ' '.join([item['prompt'] + item['completion'] for item in data])

        # 如果tokenizer是新建的，就训练并保存
        if not os.path.exists(self.tokenizer_path):
            self.tokenizer.train_from_iterator(training_text.split('\n'), self.config.model.vocab_size, min_freq=2)
            self.tokenizer.save(self.tokenizer_path)

        # 创建自定义数据集并初始化DataLoader
        self.dataset = LLMDataset(self.tokenizer, self.config.training.data_path)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=LLMDataset.collate_fn,
            num_workers=self.config.training.num_workers,  # Dynamic number of workers
            drop_last=True,
            persistent_workers=True  # Keeps workers alive for faster loading in large datasets
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        accumulation_steps = 4  # Gradually accumulate gradients over smaller batches
        progress_bar = tqdm(self.loader, desc="Training", total=len(self.loader))

        for batch_idx, batch in enumerate(progress_bar):
            self.optim.zero_grad()

            with autocast():
                inputs = batch['input_ids'].to(self.device, non_blocking=True)
                targets = batch['target_ids'].to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs.view(-1, self.config.model.vocab_size), targets.view(-1))

            self.scaler.scale(loss).backward()

            # Perform the optimizer step every 'accumulation_steps' batches
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

            # Clear memory cache
            torch.cuda.empty_cache()

        return total_loss / len(self.loader)

    def save_model(self, path):
        # Save model and tokenizer with extra safety checks
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_state_dict': self.tokenizer,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def train(self):
        # 开始训练
        for epoch in range(self.config.training.epochs):
            loss = self.train_epoch()
            print(f"Epoch {epoch} Loss: {loss:.4f}")

            # 每隔指定epoch保存模型
            if epoch % self.config.training.save_interval == 0:
                self.save_model(f"{self.config.training.save_dir}/model_{epoch}.pth")

if __name__ == "__main__":
    trainer = Trainer("config.json")
    trainer.load_data()
    trainer.train()
