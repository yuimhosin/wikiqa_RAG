# model_manager.py - 统一的模型管理器

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    统一管理嵌入模型，避免重复加载
    """
    _instance = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self, model_name="BAAI/bge-small-en-v1.5"):
        """
        获取嵌入模型单例
        """
        if self._embedding_model is None:
            print(f"初始化嵌入模型: {model_name}")
            try:
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # 明确指定设备
                    encode_kwargs={'normalize_embeddings': True}  # 标准化嵌入
                )
                print("嵌入模型初始化成功")
            except Exception as e:
                logger.error(f"嵌入模型初始化失败: {e}")
                raise
        else:
            print("使用已缓存的嵌入模型")

        return self._embedding_model

    def clear_cache(self):
        """
        清除模型缓存
        """
        self._embedding_model = None
        print("模型缓存已清除")

# 全局模型管理器实例
model_manager = ModelManager()