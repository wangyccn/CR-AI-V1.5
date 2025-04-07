"""数据集加载与预处理模块

提供以下功能:
1. 数据验证(_validate_data): 检查数据格式和类型
2. 数据清洗(_clean_data): 自动清理无效数据和特殊字符
3. 数据集类(LLMDataset): 加载、预处理和缓存数据集
4. 批处理函数(collate_fn): 处理批量数据填充和mask

支持特性:
- 文本数据处理
- 多模态支持(图像+文本)
- 数据缓存优化
- 自动截断和填充
"""

import json
import re  # 添加re模块导入
import torch
from torch.utils.data import Dataset
from torchvision import transforms  # 添加transforms导入
from PIL import Image  # 添加这行导入

def _validate_data(data):
    """
    验证数据格式
    :param data: 数据
    """
    if not data:
        raise ValueError("输入数据为空，请检查数据文件")

    required_keys = ['prompt', 'completion']
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"数据项 {idx} 不是字典类型")
        if not all(k in item for k in required_keys):
            raise ValueError(f"数据项 {idx} 缺少必要字段: {required_keys}")
        if not isinstance(
                item['prompt'],
                str) or not isinstance(
                item['completion'],
                str):
            raise TypeError(f"数据项 {idx} 的prompt或completion不是字符串类型")


class LLMDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            json_path,
            max_length=2048,
            split_ratio=0.9,
            mode='train'):
        """优化后的初始化方法"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # 初始化缓存
        self.cache = {}

        # 使用内存映射方式加载大JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        _validate_data(raw_data)

        # 更智能的数据分割
        split_idx = int(len(raw_data) * split_ratio)
        self.data = raw_data[:split_idx] if mode == 'train' else raw_data[split_idx:]

        # 使用LRU缓存提高性能
        from functools import lru_cache
        self._process_text_cached = lru_cache(
            maxsize=10000)(self._process_text)

        # 多模态支持增强
        self.multimodal = hasattr(self.tokenizer,
                                  'multimodal') and self.tokenizer.multimodal
        if self.multimodal:
            self.image_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # 图像预加载
            self.image_cache = {}

    def _load_image(self, path):
        """缓存图像加载"""
        if path not in self.image_cache:
            img = Image.open(path).convert('RGB')
            self.image_cache[path] = img
        return self.image_cache[path]

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
            return tokens[:self.max_length // 2] + \
                tokens[-self.max_length // 2:]

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

        # 分别处理输入和目标
        input_text = item['prompt']
        target_text = item['completion']

        # 编码文本
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)

        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        if len(target_ids) > self.max_length:
            target_ids = target_ids[:self.max_length]

        # 转换为torch tensor
        result = {
            'input_ids': torch.LongTensor(input_ids),
            'target_ids': torch.LongTensor(target_ids),
            'prompt': input_text,
            'completion': target_text
        }

        # 添加多模态支持
        if 'image_path' in item and hasattr(self, 'image_transform'):
            try:
                from PIL import Image
                image = Image.open(item['image_path']).convert('RGB')
                result['images'] = self.image_transform(image)
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

        # 获取最大长度
        max_input_len = max(len(seq) for seq in inputs)
        max_target_len = max(len(seq) for seq in targets)

        # 创建填充后的tensor
        padded_inputs = torch.zeros(
            (len(batch), max_input_len), dtype=torch.long)
        padded_targets = torch.zeros(
            (len(batch), max_target_len), dtype=torch.long)
        attention_mask = torch.zeros(
            (len(batch), max_input_len), dtype=torch.float)

        # 填充数据
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            input_length = len(inp)
            target_length = len(tgt)

            padded_inputs[i, :input_length] = inp
            padded_targets[i, :target_length] = tgt
            attention_mask[i, :input_length] = 1.0

        result = {
            'input_ids': padded_inputs,
            'target_ids': padded_targets,
            'attention_mask': attention_mask
        }

        # 处理图像数据
        if 'images' in batch[0]:
            images = [item.get('images') for item in batch]
            if all(img is not None for img in images):
                result['images'] = torch.stack(images)

        return result
