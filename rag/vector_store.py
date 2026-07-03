# rag/vector_store.py
import os
import numpy as np
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from model.factory import embed_model
from sklearn.metrics.pairwise import cosine_similarity


class VectorStoreService:
    """向量存储服务 - 使用内存模式避免文件锁定"""

    def __init__(self):
        """初始化向量存储"""
        self.documents = []
        self.embeddings = []
        print("[OK] ChromaDB 内存模式启动成功 (不持久化)")

    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储"""
        if not documents:
            print("[WARN] 没有文档需要添加")
            return

        try:
            for doc in documents:
                # 生成文档的嵌入向量
                embedding = embed_model.embed_query(doc.page_content)
                self.documents.append(doc)
                self.embeddings.append(np.array(embedding))

            print(f"[OK] 成功添加 {len(documents)} 个文档到内存")
        except Exception as e:
            print(f"[ERROR] 添加文档失败: {e}")
            raise

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        获取检索器

        Args:
            search_kwargs: 检索参数，如 {"k": 4}
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        k = search_kwargs.get("k", 4)

        class SimpleRetriever:
            def __init__(self, parent, top_k):
                self.parent = parent
                self.top_k = top_k

            def invoke(self, query: str):
                """执行检索"""
                return self.parent.similarity_search(query, self.top_k)

        return SimpleRetriever(self, k)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not self.documents:
            print("[WARN] 向量存储为空，请先添加文档")
            return []

        try:
            # 生成查询嵌入
            query_embedding = np.array(embed_model.embed_query(query)).reshape(1, -1)

            # 检查是否有嵌入
            if not self.embeddings:
                return []

            # 转换嵌入矩阵
            embeddings_matrix = np.array(self.embeddings)

            # 计算余弦相似度
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]

            # 获取 top k 索引
            k = min(k, len(similarities))
            top_indices = similarities.argsort()[-k:][::-1]

            # 返回文档
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    results.append(self.documents[idx])

            print(f"[OK] 检索到 {len(results)} 个相关文档")
            return results

        except Exception as e:
            print(f"[ERROR] 相似性搜索失败: {e}")
            return []

    def delete_all(self):
        """删除所有文档"""
        self.documents = []
        self.embeddings = []
        print("[INFO] 已清空所有文档")

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)


# 创建全局实例（延迟初始化）
_vector_store = None


def get_vector_store() -> VectorStoreService:
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store