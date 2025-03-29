import json
import re  # 添加re模块导入
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # 添加transforms导入

def _validate_data(data):
    """
    验证数据格式
    :param data: 数据
    """
    required_keys = ['prompt', 'completion']
    for idx, item in enumerate(data):
        if not all(k in item for k in required_keys):
            raise ValueError(f"Invalid data format at index {idx}")
        if not isinstance(item['prompt'], str) or not isinstance(item['completion'], str):
            raise TypeError(f"Non-string values at index {idx}")


class LLMDataset(Dataset):
    def __init__(self, tokenizer, json_path, max_length=2048, split_ratio=0.9, mode='train'):
        """
        初始化数据集，加载JSON数据，进行预处理，并拆分为训练和验证集
        :param tokenizer: 分词器
        :param json_path: JSON文件路径
        :param max_length: 最大序列长度
        :param split_ratio: 数据集划分比例（训练集/验证集）
        :param mode: 模式（训练集/验证集）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取并验证数据
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        _validate_data(raw_data)

        split_idx = int(len(raw_data) * split_ratio)
        self.data = raw_data[:split_idx] if mode == 'train' else raw_data[split_idx:]

        # 使用字典缓存已处理的输入
        self.cache = {}

        # 添加多模态支持
        self.multimodal = hasattr(self.tokenizer, 'multimodal') and self.tokenizer.multimodal
        if self.multimodal:
            self.image_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _process_text(self, text):
        """
        处理文本并编码为token ID
        :param text: 输入文本
        :return: 编码后的token列表
        """
        tokens = self.tokenizer.encode(text)
        return tokens

    def _truncate_pad(self, tokens):
        """
        截断和填充token序列
        :param tokens: 输入tokens
        :return: 截断或填充后的token序列
        """
        if len(tokens) > self.max_length:
            return tokens[:self.max_length // 2] + tokens[-self.max_length // 2:]

        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        return tokens + [pad_token_id] * (self.max_length - len(tokens))

    def __len__(self):
        """
        返回数据集的大小
        :return: 数据集长度
        """
        return len(self.data)

    def _clean_data(self, data):
        """自动数据清洗"""
        cleaned = []
        for item in data:
            # 去除空文本和无效数据
            if not item['prompt'] or not item['completion']:
                continue
                
            # 去除HTML标签
            item['prompt'] = re.sub(r'<[^>]+>', '', item['prompt'])
            item['completion'] = re.sub(r'<[^>]+>', '', item['completion'])
            
            # 去除特殊字符
            item['prompt'] = item['prompt'].strip()
            item['completion'] = item['completion'].strip()
            
            cleaned.append(item)
        return cleaned

    def __getitem__(self, idx):
        # 检查缓存中是否有已处理的结果
        if idx in self.cache:
            return self.cache[idx]

        item = self.data[idx]
        # 合并问题和答案文本
        full_text = f"{item['prompt']} {item['completion']} <eos>"

        # 编码文本
        tokens = self._process_text(full_text)
        input_ids = self._truncate_pad(tokens[:-1])  # 确保这行存在
        target_ids = self._truncate_pad(tokens[1:])  # 确保这行存在

        # 转换为torch tensor - 修改这里，不要使用unsqueeze(0)
        result = {
            'input_ids': torch.LongTensor(input_ids),
            'target_ids': torch.LongTensor(target_ids)
        }

        # 添加多模态支持
        if hasattr(self, 'multimodal') and self.multimodal and 'image_path' in item:
            try:
                image = self._load_image(item['image_path'])
                result['images'] = image
            except Exception as e:
                print(f"加载图像失败: {e}")
                # 回退到纯文本模式
                pass

        # 缓存处理后的结果
        self.cache[idx] = result
        return result

    @staticmethod
    def collate_fn(batch):
        """
        合并批量数据，处理填充和mask
        :param batch: 数据批量(字典列表)
        :return: 字典，包含input_ids, target_ids, attention_mask
        """
        # 从每个样本字典中提取input_ids和target_ids
        inputs = [item['input_ids'] for item in batch]
        targets = [item['target_ids'] for item in batch]

        max_len = max(len(seq) for seq in inputs)

        # 创建填充后的tensor
        padded_inputs = torch.full((len(batch), max_len), 0, dtype=torch.long)
        padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.float)

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            length = len(inp)
            padded_inputs[i, :length] = inp
            padded_targets[i, :length] = tgt[:max_len]
            attention_mask[i, :length] = 1.0

        return {
            'input_ids': padded_inputs,
            'target_ids': padded_targets,
            'attention_mask': attention_mask
        }