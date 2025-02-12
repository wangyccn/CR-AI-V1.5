from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from typing import List


class BpeTokenizer:
    def __init__(self, path=None):
        """
        初始化BPE分词器，配置正常化、预处理、解码、后处理等步骤
        :param path: 如果提供路径，将加载已经存在的分词器文件
        """
        self._special_tokens = ["<s>", "</s>", "<|user|>", "<|system|>", "<pad>"]
        self._tokenizer = Tokenizer(BPE())

        # 配置分词器的正常化、预处理、解码和后处理器
        self._configure_tokenizer()

        # 如果提供路径，加载已训练好的分词器
        if path:
            self.load(path)

    def _configure_tokenizer(self):
        """
        配置分词器的正常化、预处理、解码和后处理器
        """
        # 词典正则化步骤
        self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])  # 统一字符规范化

        # 预处理器
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(behavior="isolated"),  # 分割标点符号
            pre_tokenizers.Metaspace(prepend_scheme="never"),  # 在字符间添加空格
            pre_tokenizers.Split(pattern=r"(\d+|[a-zA-Z]+|(?:'s|'t|'re|'ve|'m|'ll|'d))", behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)  # 字节级处理
        ])

        # 解码器配置
        self._tokenizer.decoder = decoders.Sequence([
            decoders.ByteLevel(),  # 使用ByteLevel解码器
            decoders.Metaspace(prepend_scheme="never")  # 配置解码时空格的处理方式
        ])

        # 后处理器
        self._tokenizer.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False)])  # 保持偏移量完整

    def _init_trainer(self, vocab_size: int, min_freq: int) -> BpeTrainer:
        """
        初始化分词训练器，设置词汇表的最大大小和最小频率
        :param vocab_size: 词汇表大小
        :param min_freq: 最小频率
        :return: BpeTrainer对象
        """
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size = len(self._special_tokens) + len(alphabet)  # 加入特殊标记
        assert vocab_size > min_size, "vocab_size必须大于特殊标记和字母表大小之和"

        lim_len = vocab_size - len(self._special_tokens)
        return BpeTrainer(
            vocab_size=lim_len,
            min_frequency=min_freq,
            limit_alphabet=vocab_size // 4,
            max_token_length=12,
            show_progress=True,
            initial_alphabet=alphabet,
        )

    def train(self, files: List[str], vocab_size: int, min_freq: int):
        """
        使用文件训练BPE分词器
        :param files: 训练文本文件列表
        :param vocab_size: 词汇表大小
        :param min_freq: 最小词频
        """
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train(files=files, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens)

    def train_from_iterator(self, iterator, vocab_size: int, min_freq: int):
        """
        使用迭代器训练BPE分词器
        :param iterator: 训练数据的迭代器
        :param vocab_size: 词汇表大小
        :param min_freq: 最小词频
        """
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        self._tokenizer.add_special_tokens(self._special_tokens)

    def save(self, path: str):
        """
        保存训练好的分词器
        :param path: 保存路径
        """
        self._tokenizer.save(path)

    def load(self, path: str):
        """
        加载保存的分词器
        :param path: 分词器文件路径
        """
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, tokens: str, out_ids=True) -> List:
        """
        编码文本为token ID列表
        :param tokens: 输入文本
        :param out_ids: 是否返回token ID，默认为True
        :return: token ID列表或token列表
        """
        encoded: Encoding = self._tokenizer.encode(tokens)
        return encoded.ids if out_ids else encoded.tokens

    def decode(self, tokens: List[int]) -> str:
        """
        解码token ID为原始文本
        :param tokens: token ID列表
        :return: 解码后的文本
        """
        return self._tokenizer.decode(tokens)

    def __len__(self) -> int:
        """
        获取词汇表大小
        :return: 词汇表大小
        """
        return self._tokenizer.get_vocab_size()

    @property
    def stop_ids(self) -> List[int]:
        """
        获取特殊标记的token ID
        :return: 特殊标记的token ID列表
        """
        return [self._tokenizer.token_to_id(token) for token in self._special_tokens]

    def save(self, path: str):
        """
        Save the tokenizer with additional checks
        :param path: 保存路径
        """
        self._tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """
        Load a saved tokenizer from file
        :param path: Tokenizer file path
        """
        self._tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path}")