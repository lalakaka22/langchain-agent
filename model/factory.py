# model/factory.py
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv

# Windows 下修复 stdout/stderr 编码，避免 GBK 无法编码特殊字符
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 尝试导入新版本 Ollama
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    OLLAMA_NEW = True
    print("[OK] 使用 langchain-ollama (推荐)")
except ImportError:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

    OLLAMA_NEW = False
    print("[WARN] 使用 langchain-community (建议安装 langchain-ollama)")

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# 加载环境变量
load_dotenv()


class BaseModelFactory(ABC):
    """模型工厂基类"""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂"""

    def generator(self) -> Optional[BaseChatModel]:
        """创建聊天模型"""
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()

        if provider == "ollama":
            return self._create_ollama_chat()
        elif provider == "tongyi":
            return self._create_tongyi_chat()
        else:
            raise ValueError(f"不支持的模型提供商: {provider}")

    def _create_ollama_chat(self):
        """创建 Ollama 聊天模型"""
        model_name = os.getenv("OLLAMA_CHAT_MODEL", "qwen:34b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"> 初始化 Ollama Chat 模型: {model_name}")
        print(f"> Ollama 服务地址: {base_url}")

        # 测试 Ollama 连接
        try:
            import requests
            response = requests.get(f"{base_url}/api/version", timeout=3)
            if response.status_code == 200:
                print("[OK] Ollama 服务连接成功")
            else:
                print(f"[WARN] Ollama 服务响应异常: {response.status_code}")
        except Exception as e:
            print(f"[WARN] 无法连接到 Ollama 服务: {e}")
            print("请确保运行: ollama serve")

        # 创建模型
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
            top_k=40,
            top_p=0.9,
        )

    def _create_tongyi_chat(self):
        """创建通义千问聊天模型"""
        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            raise ImportError("请安装 langchain-community: pip install langchain-community")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        return ChatTongyi(
            model=os.getenv("TONGYI_MODEL", "qwen-plus"),
            dashscope_api_key=api_key,
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )


class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂"""

    def generator(self) -> Optional[Embeddings]:
        """创建嵌入模型"""
        provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

        if provider == "ollama":
            return self._create_ollama_embeddings()
        elif provider == "tongyi":
            return self._create_tongyi_embeddings()
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")

    def _create_ollama_embeddings(self):
        """创建 Ollama 嵌入模型"""
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"> 初始化 Ollama Embeddings 模型: {model_name}")

        return OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
        )

    def _create_tongyi_embeddings(self):
        """创建通义千问嵌入模型"""
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError:
            raise ImportError("请安装 langchain-community: pip install langchain-community")

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        return DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
            dashscope_api_key=api_key,
        )


# 创建全局实例
try:
    chat_model = ChatModelFactory().generator()
    embed_model = EmbeddingsFactory().generator()
    print("[OK] 模型初始化完成")
except Exception as e:
    print(f"[ERROR] 模型初始化失败: {e}")
    raise