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
        """初始化BPE分词器"""
        self._special_tokens = ["<s>", "</s>", "<|user|>", "<|system|>", "<pad>", "<unk>"]
        try:
            self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self._tokenizer.add_special_tokens(self._special_tokens)
            
            # 确保这些属性在初始化时设置
            self.eos_token = "</s>"
            self.eos_token_id = None  # 会在_init_special_tokens中设置
            
            self._init_special_tokens()
            self._configure_tokenizer()

            if path:
                self.load(path)
                
        except Exception as e:
            logger.error(f"初始化tokenizer失败: {str(e)}")
            raise RuntimeError("Tokenizer初始化失败") from e

    def _init_special_tokens(self) -> None:
        """初始化并验证特殊标记"""
        try:
            for token in self._special_tokens:
                if self._tokenizer.token_to_id(token) is None:
                    # 如果标记不存在，先添加到词汇表
                    self._tokenizer.add_tokens([token])
                    logger.warning(f"特殊标记 {token} 不在词汇表中，已自动添加")
            
            # 确保这些属性被正确设置
            self.pad_token = "<pad>"
            self.unk_token = "<unk>"
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            
            self.pad_token_id = self._tokenizer.token_to_id(self.pad_token)
            self.unk_token_id = self._tokenizer.token_to_id(self.unk_token)
            self.bos_token_id = self._tokenizer.token_to_id(self.bos_token)
            self.eos_token_id = self._tokenizer.token_to_id(self.eos_token)
            
        except Exception as e:
            logger.error(f"初始化特殊标记失败: {str(e)}")
            raise RuntimeError("特殊标记初始化失败") from e

    def load(self, path: str) -> None:
        """从指定路径加载分词器"""
        try:
            self._tokenizer = Tokenizer.from_file(path)
            # 确保特殊标记存在
            self._tokenizer.add_special_tokens(self._special_tokens)
            self._init_special_tokens()
            self._configure_tokenizer()
            logger.info(f"分词器已从 {path} 加载")

        except Exception as e:
            logger.error(f"加载分词器时出错: {str(e)}")
            raise RuntimeError("无法加载分词器") from e

    def encode(self, text: Union[str, List[str]], out_ids: bool = True) -> Union[List[int], List[str], List[List[int]], List[List[str]]]:
        """
        增强的编码方法，支持单个文本和文本列表
        """
        if isinstance(text, list):
            return [self._encode_single(t, out_ids) for t in text]
        return self._encode_single(text, out_ids)
    
    def _encode_single(self, text: str, out_ids: bool = True) -> Union[List[int], List[str]]:
        """单个文本的编码实现"""
        if not isinstance(text, str):
            text = str(text)
            
        try:
            encoded: Encoding = self._tokenizer.encode(text)
            if not encoded.ids and not encoded.tokens:
                logger.warning(f"文本 '{text}' 编码结果为空")
                return [] if out_ids else []
            return encoded.ids if out_ids else encoded.tokens
        except Exception as e:
            logger.error(f"编码文本时出错: {str(e)}")
            return [] if out_ids else []

    def decode(self, tokens: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        增强的解码方法，支持单个序列和序列列表
        """
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            return [self._decode_single(t) for t in tokens]
        return self._decode_single(tokens)
    
    def _decode_single(self, tokens: List[int]) -> str:
        """单个序列的解码实现"""
        try:
            if not tokens:
                return ""
            result = self._tokenizer.decode(tokens)
            return result if result else ""
        except Exception as e:
            logger.error(f"解码token时出错: {str(e)}")
            return ""

    def _configure_tokenizer(self) -> None:
        """配置分词器的正常化、预处理、解码和后处理步骤"""
        try:
            # 配置正则化器
            self._tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(),  # Unicode 正规化
                normalizers.StripAccents(),  # 去除重音符号
                normalizers.Replace(r'[\s]+', ' '),  # 合并多个空格
            ])

            # 配置预分词器
            self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),  # 按空格分词
                pre_tokenizers.Punctuation(),  # 处理标点符号
            ])

            # 配置解码器
            self._tokenizer.decoder = decoders.ByteLevel()

            # 配置后处理器
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{self.bos_token} $A {self.eos_token}",
                pair=f"{self.bos_token} $A {self.eos_token} $B:1 {self.eos_token}:1",
                special_tokens=[
                    (self.bos_token, self.bos_token_id),
                    (self.eos_token, self.eos_token_id),
                ],
            )
            
            logger.info("分词器配置完成")
            
        except Exception as e:
            logger.error(f"配置分词器时出错: {str(e)}")
            raise

    def train_from_iterator(self, iterator: Iterable[str], vocab_size: int, min_freq: int = 2) -> None:
        """从文本迭代器训练BPE分词器"""
        try:
            # 创建BPE训练器
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_freq,
                special_tokens=self._special_tokens,
                show_progress=True
            )
            
            # 训练分词器
            self._tokenizer.train_from_iterator(iterator, trainer=trainer)
            
            # 重新初始化特殊token映射
            self._init_special_tokens()
            
            logger.info(f"训练完成，词汇量: {vocab_size}")
            
        except Exception as e:
            logger.error(f"训练分词器时出错: {str(e)}")
            raise RuntimeError("分词器训练失败") from e

    def save(self, path: str) -> None:
        """保存分词器到指定路径"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存分词器
            self._tokenizer.save(path)
            logger.info(f"分词器已保存到 {path}")
            
        except Exception as e:
            logger.error(f"保存分词器时出错: {str(e)}")
            raise RuntimeError(f"无法保存分词器到 {path}") from e

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