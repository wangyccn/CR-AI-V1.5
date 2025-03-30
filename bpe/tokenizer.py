from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from typing import List, Optional, Union, Iterable
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BpeTokenizer:
    def __init__(self, path: Optional[str] = None):
        """
        初始化BPE分词器，配置正常化、预处理、解码、后处理等步骤
        :param path: 如果提供路径，将加载已经存在的分词器文件
        """
        self._special_tokens = ["<s>", "</s>", "<|user|>", "<|system|>", "<pad>", "<unk>"]
        self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # 添加特殊token的ID映射
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        # 初始化时立即添加特殊标记以确保ID存在
        self._tokenizer.add_special_tokens(self._special_tokens)
        
        # 设置特殊token的ID
        self.pad_token_id = self._tokenizer.token_to_id("<pad>")
        self.unk_token_id = self._tokenizer.token_to_id("<unk>")
        self.bos_token_id = self._tokenizer.token_to_id("<s>")
        self.eos_token_id = self._tokenizer.token_to_id("</s>")

        # 配置分词器的正常化、预处理、解码和后处理器
        self._configure_tokenizer()

        # 如果提供路径，加载已训练好的分词器
        if path:
            self.load(path)

    def _configure_tokenizer(self) -> None:
        """
        配置分词器的正常化、预处理、解码和后处理器
        """
        try:
            # 正规化器配置 - 使用正确的NFC实例化方法
            self._tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFC(),
                normalizers.Replace(" ", "▁")  # 修改为正确的参数传递方式
            ])
            
            # 预分词器配置
            self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(
                    pattern=r"(\d+|[a-zA-Z]+|(?:'s|'t|'re|'ve|'m|'ll|'d))",
                    behavior="isolated"
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False)
            ])
            
            # 解码器配置
            self._tokenizer.decoder = decoders.Sequence([
                decoders.ByteLevel(),
                decoders.Replace("▁", " ")  # 修改为正确的参数传递方式
            ])
            
            # 后处理器配置 - 优化特殊标记处理
            special_tokens_with_ids = []
            required_tokens = ["<s>", "</s>"]
            missing_tokens = []
            
            # 检查关键特殊标记
            for token in self._special_tokens:
                token_id = self._tokenizer.token_to_id(token)
                if token_id is not None:
                    special_tokens_with_ids.append((token, token_id))
                elif token in required_tokens:
                    missing_tokens.append(token)
            
            # 检查是否有必要的特殊标记
            if not missing_tokens and len(special_tokens_with_ids) >= 2:
                self._tokenizer.post_processor = processors.TemplateProcessing(
                    single="<s> $A </s>",
                    pair="<s> $A </s> $B </s>",  # 添加对文本对的支持
                    special_tokens=special_tokens_with_ids
                )
            else:
                logger.warning(f"特殊标记缺失: {missing_tokens}，跳过后处理器配置")
                
        except Exception as e:
            logger.error(f"配置tokenizer时出错: {str(e)}")
            # 记录更详细的错误信息以便调试
            import traceback
            logger.debug(f"错误详情: {traceback.format_exc()}")
            raise

    def _init_trainer(self, vocab_size: int, min_freq: int) -> BpeTrainer:
        """
        初始化分词训练器，设置词汇表的最大大小和最小频率
        :param vocab_size: 词汇表大小
        :param min_freq: 最小频率
        :return: BpeTrainer对象
        """
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size = len(self._special_tokens) + len(alphabet)
        if vocab_size <= min_size:
            raise ValueError(f"vocab_size必须大于{min_size}")

        return BpeTrainer(
            vocab_size=vocab_size - len(self._special_tokens),
            min_frequency=min_freq,
            special_tokens=self._special_tokens,
            initial_alphabet=alphabet,
            show_progress=True
        )

    def train(self, files: List[str], vocab_size: int, min_freq: int) -> None:
        """
        使用文件训练BPE分词器
        :param files: 训练文本文件列表
        :param vocab_size: 词汇表大小
        :param min_freq: 最小词频
        """
        try:
            trainer = self._init_trainer(vocab_size, min_freq)
            self._tokenizer.train(files=files, trainer=trainer)
            logger.info("Training completed successfully")
            # 重新配置以更新后处理器中的特殊标记ID
            self._configure_tokenizer()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def train_from_iterator(self, iterator: Iterable[str], vocab_size: int, min_freq: int) -> None:
        """
        使用迭代器训练BPE分词器
        :param iterator: 训练数据的迭代器
        :param vocab_size: 词汇表大小
        :param min_freq: 最小词频
        """
        try:
            trainer = self._init_trainer(vocab_size, min_freq)
            self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
            logger.info("Training from iterator completed successfully")
            self._configure_tokenizer()  # 更新后处理器
        except Exception as e:
            logger.error(f"Training from iterator failed: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """
        保存训练好的分词器
        :param path: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._tokenizer.save(path)
            logger.info(f"Tokenizer saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {str(e)}")
            raise

    def load(self, path: str) -> 'BpeTokenizer':
        """
        加载保存的分词器
        :param path: 分词器文件路径
        :return: 加载后的分词器实例
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tokenizer file not found at {path}")
                
            self._tokenizer = Tokenizer.from_file(path)
            # 确保特殊标记存在
            self._tokenizer.add_special_tokens(self._special_tokens)
            self._configure_tokenizer()
            
            # 验证tokenizer是否有效
            test_text = "Hello world"
            test_tokens = self._tokenizer.encode(test_text)
            if not test_tokens:
                raise ValueError("无法使用加载的tokenizer编码文本")
                
            logger.info(f"Tokenizer loaded from {path}")
            return self
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

    def encode(self, text: str, out_ids: bool = True) -> Union[List[int], List[str]]:
        """
        编码文本为token ID列表或token列表
        :param text: 输入文本
        :param out_ids: 是否返回token ID，默认为True
        :return: token ID列表或token列表
        """
        if not isinstance(text, str):
            text = str(text)
            
        try:
            encoded: Encoding = self._tokenizer.encode(text)
            return encoded.ids if out_ids else encoded.tokens
        except Exception as e:
            logger.error(f"编码文本时出错: {str(e)}")
            # 如果编码失败，返回空列表而不是抛出异常
            return [] if out_ids else []

    def decode(self, tokens: List[int]) -> str:
        """
        解码token ID为原始文本
        :param tokens: token ID列表
        :return: 解码后的文本
        """
        try:
            return self._tokenizer.decode(tokens)
        except Exception as e:
            logger.error(f"解码token时出错: {str(e)}")
            return ""

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
        result = []
        for token in self._special_tokens:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is not None:
                result.append(token_id)
        return result